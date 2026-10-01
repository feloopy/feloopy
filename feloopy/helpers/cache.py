# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import os
import time
import pickle
import atexit
import collections


class _PersistentLRUCache:
    """LRU cache with disk persistence and automatic cleanup."""

    _CACHE_DIR = os.path.join(os.path.expanduser('~'), 'feloopy', 'Caches', 'API')
    _MAX_SIZE_MB = 256
    _MAX_AGE_DAYS = 30
    _MAX_ENTRIES = 512

    def __init__(self, maxsize=512):
        self._cache = collections.OrderedDict()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0
        self._dirty = False
        self._loaded = False
        self._ensure_cache_dir()
        atexit.register(self._save_to_disk)

    def _ensure_cache_dir(self):
        os.makedirs(self._CACHE_DIR, exist_ok=True)

    def _cache_file(self):
        return os.path.join(self._CACHE_DIR, 'model_cache.pkl')

    def _load_from_disk(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            path = self._cache_file()
            if os.path.exists(path):
                age_days = (time.time() - os.path.getmtime(path)) / 86400
                if age_days > self._MAX_AGE_DAYS:
                    os.remove(path)
                    return
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and data.get('version') == 2:
                    now = time.time()
                    entries = {}
                    for k, v in data.get('entries', {}).items():
                        if isinstance(v, dict):
                            ts = v.get('timestamp')
                            if ts is not None and (now - ts) / 86400 > self._MAX_AGE_DAYS:
                                continue  # entry outlived its usefulness
                        entries[k] = v
                    self._cache = collections.OrderedDict(entries)
                    self._maxsize = min(self._maxsize, data.get('max_entries', self._maxsize))
                    self._trim_to_size()
                else:
                    os.remove(path)
        except Exception:
            try:
                os.remove(path)
            except Exception:
                pass
            self._cache = collections.OrderedDict()

    def _trim_to_size(self):
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
            self._dirty = True

    def _save_to_disk(self):
        if not self._dirty or not self._loaded:
            return
        try:
            entries = {}
            for k, v in self._cache.items():
                if isinstance(v, dict):
                    entry = {}
                    for ek, ev in v.items():
                        if ek in ('em', 'model'):
                            continue
                        try:
                            pickle.dumps(ev)
                            entry[ek] = ev
                        except Exception:
                            continue
                    # keep the original creation time so per-entry expiry
                    # in _load_from_disk stays meaningful
                    entry.setdefault('timestamp', time.time())
                    entries[k] = entry
                else:
                    entries[k] = v
            data = {
                'version': 2,
                'entries': entries,
                'max_entries': self._maxsize,
            }
            path = self._cache_file()
            tmp_path = path + '.tmp'
            with open(tmp_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp_path, path)
            self._dirty = False
            self._enforce_disk_limit()
        except Exception:
            pass

    def _enforce_disk_limit(self):
        try:
            path = self._cache_file()
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                if size_mb > self._MAX_SIZE_MB:
                    entries = list(self._cache.items())
                    keep = len(entries) // 2
                    self._cache = collections.OrderedDict(entries[keep:])
                    self._dirty = True
                    self._save_to_disk()
        except Exception:
            pass

    def get(self, key, default=None):
        self._load_from_disk()
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return default

    def __contains__(self, key):
        self._load_from_disk()
        return key in self._cache

    def __setitem__(self, key, value):
        self._load_from_disk()
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        self._dirty = True
        self._trim_to_size()
        if len(self._cache) % 10 == 0:
            self._save_to_disk()

    def __len__(self):
        self._load_from_disk()
        return len(self._cache)

    def pop(self, key, *args):
        self._load_from_disk()
        if key in self._cache:
            self._dirty = True
        return self._cache.pop(key, *args)

    def clear(self):
        self._load_from_disk()
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._dirty = False
        try:
            path = self._cache_file()
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    def keys(self):
        self._load_from_disk()
        return list(self._cache.keys())

    def items(self):
        self._load_from_disk()
        return list(self._cache.items())

    def stats(self):
        self._load_from_disk()
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        disk_size_mb = 0
        try:
            path = self._cache_file()
            if os.path.exists(path):
                disk_size_mb = os.path.getsize(path) / (1024 * 1024)
        except Exception:
            pass
        return {
            'size': len(self._cache),
            'maxsize': self._maxsize,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'disk_size_mb': int(disk_size_mb * 100) / 100 if disk_size_mb else 0,
            'cache_dir': self._CACHE_DIR,
        }

    def invalidate_stale(self, environment=None):
        """Drop entries whose stored function metadata no longer matches.

        With ``environment``, entries recorded for that function name whose
        bytecode or constants differ from the current function are removed
        (classic stale-cache invalidation).  Without it, only entries that
        carry no function metadata at all -- and therefore can never be
        validated -- are removed.

        Returns the number of entries removed.
        """
        self._load_from_disk()
        target = None
        if environment is not None and hasattr(environment, '__code__'):
            code = environment.__code__
            target = (environment.__name__, code.co_code, code.co_consts)
        stale_keys = []
        for key, entry in self._cache.items():
            if not isinstance(entry, dict) or not entry.get('func_code'):
                if target is None:
                    stale_keys.append(key)
                continue
            if target is None:
                continue
            stored = (entry.get('func_name'), entry.get('func_code'),
                      entry.get('func_consts'))
            if stored[0] == target[0] and stored[1:] != target[1:]:
                stale_keys.append(key)
        for key in stale_keys:
            self._cache.pop(key, None)
            self._dirty = True
        if stale_keys:
            self._save_to_disk()
        return len(stale_keys)
