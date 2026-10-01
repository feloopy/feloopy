# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""pip-style release notifier that can never generate meaningful PyPI traffic.

Environment variables
---------------------
``FELOOPY_DISABLE_UPDATE_CHECK``  set to 1/true/yes/on to disable entirely.
``FELOOPY_UPDATE_CHECK``          1/0 overrides every automatic rule.
``FELOOPY_UPDATE_CHECK_INTERVAL_DAYS``  override the throttling interval.
``FELOOPY_UPDATE_STATE_FILE``     override the state file location (tests/CI).
``NO_COLOR``                      disable colouring of the notice.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request

from .._version import __version__

__all__ = ["check_update", "schedule_update_check"]

# ---------------------------------------------------------------------------
# Configuration (constants are module level so tests can patch them)
# ---------------------------------------------------------------------------

PYPI_URL = "https://pypi.org/pypi/feloopy/json"
UPGRADE_URL = "https://pypi.org/project/feloopy/"
RELEASES_URL = UPGRADE_URL + "#history"

DEFAULT_INTERVAL_DAYS = 7.0
MIN_INTERVAL_SECONDS = 3600.0          # never allow a tighter automatic loop
MAX_INTERVAL_DAYS = 365.0
JITTER_SECONDS = 24 * 60 * 60          # +/- 1 day
TIMEOUT_SECONDS = 3.0
MAX_RESPONSE_BYTES = 1_000_000
MAX_BACKOFF_DAYS = 30
LOCK_STALE_SECONDS = 600
STATE_SCHEMA = 1

ENV_DISABLE = "FELOOPY_DISABLE_UPDATE_CHECK"
ENV_ENABLE = "FELOOPY_UPDATE_CHECK"
ENV_INTERVAL = "FELOOPY_UPDATE_CHECK_INTERVAL_DAYS"
ENV_STATE_FILE = "FELOOPY_UPDATE_STATE_FILE"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

_VERSION_PATTERN = re.compile(r"[0-9A-Za-z][0-9A-Za-z._+-]{0,63}")

_INFINITE = 10 ** 9
_MAX_RELEASE_LEN = 8
_PRE_RANK = {"a": 0, "alpha": 0, "b": 1, "beta": 1, "c": 2, "rc": 2, "pre": 2, "preview": 2}

_scheduled = False
_schedule_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Version handling (PEP 440 subset, used only when ``packaging`` is absent)
# ---------------------------------------------------------------------------

def _parse_version(version):
    """Return a sortable key for *version*, or ``None`` if it cannot be parsed.

    Implements the ordering rules that matter for release comparison:
    epoch, release, pre-release (a/b/rc), post-release, dev-release and local
    version segments.
    """
    if not isinstance(version, str):
        return None
    text = version.strip()
    if not text or len(text) > 80:
        return None
    if text[:1] in ("v", "V"):
        text = text[1:]

    epoch = 0
    if "!" in text:
        head, _, text = text.partition("!")
        if not head.isdigit():
            return None
        epoch = int(head)

    local = ()
    if "+" in text:
        text, _, raw_local = text.partition("+")
        local = tuple(
            (0, int(part)) if part.isdigit() else (1, part.lower())
            for part in re.split(r"[._-]", raw_local) if part
        )

    match = re.match(r"^(\d+(?:\.\d+)*)(.*)$", text)
    if not match:
        return None
    release = tuple(int(p) for p in match.group(1).split("."))
    if len(release) > _MAX_RELEASE_LEN:
        return None
    release = release + (0,) * (_MAX_RELEASE_LEN - len(release))

    suffix = _parse_suffix(match.group(2))
    if suffix is None:
        return None
    pre_key, post_num, dev_num = suffix

    if pre_key is None:
        # No pre-release: dev-only releases sort before final/post releases.
        pre_key = -1 if (post_num is None and dev_num is not None) else _INFINITE
    # Absent post must sort *first* (final release precedes its .postN), while
    # absent pre/dev sort last.  This mirrors packaging.version._cmpkey.
    post_key = -1 if post_num is None else min(post_num, _INFINITE - 1)
    dev_key = _INFINITE if dev_num is None else min(dev_num, _INFINITE - 1)

    return (
        epoch, release, pre_key, post_key, dev_key,
        1 if local else 0, local,
    )


def _parse_suffix(rest):
    """Parse ``pre``/``post``/``dev`` markers from the non-numeric *rest*.

    Returns ``(pre_key, post_num, dev_num)`` (``None`` for an absent part) or
    ``None`` when the suffix does not follow the expected grammar.
    """
    text = rest.lower().lstrip("._-")
    pre = post = dev = None

    match = re.match(r"(preview|pre|alpha|beta|rc|a|b|c)(\d*)(?=[._-]|$)", text)
    if match:
        rank = _PRE_RANK.get(match.group(1))
        if rank is None:
            return None
        pre = rank * 1_000_000 + (int(match.group(2)) if match.group(2) else 0)
        text = text[match.end():].lstrip("._-")

    if text:
        match = re.match(r"(revision|rev|post|r)(\d*)(?=[._-]|$)", text)
        if match:
            post = int(match.group(2)) if match.group(2) else 0
            text = text[match.end():].lstrip("._-")

    if text:
        match = re.match(r"dev(\d*)(?=[._-]|$)", text)
        if match:
            dev = int(match.group(1)) if match.group(1) else 0
            text = text[match.end():].lstrip("._-")

    if text:
        return None
    if pre is None and post is None and dev is None:
        return None if rest.strip("._-") else (None, None, None)
    return pre, post, dev


def _compare_versions(left, right):
    """Return ``-1``/``0``/``1`` for *left* vs *right*, or ``None`` if unknown."""
    if not isinstance(left, str) or not isinstance(right, str):
        return None
    if left == right:
        return 0
    try:
        from packaging.version import Version  # preferred: exact PEP 440
        left_v, right_v = Version(left), Version(right)
    except Exception:
        left_key, right_key = _parse_version(left), _parse_version(right)
        if left_key is None or right_key is None:
            return None
        return -1 if left_key < right_key else (1 if left_key > right_key else 0)
    return -1 if left_v < right_v else (1 if left_v > right_v else 0)


def _is_newer(candidate, current):
    """True when *candidate* is a stable release strictly newer than *current*."""
    if not isinstance(candidate, str) or not _VERSION_PATTERN.fullmatch(candidate):
        return False
    if _parse_version(candidate) is None:
        return False
    if _is_prerelease(candidate) and not _is_prerelease(current):
        return False   # never push stable users onto a beta/dev release
    comparison = _compare_versions(candidate, current)
    return comparison == 1


def _is_prerelease(version):
    """True when *version* carries a pre-release or dev marker."""
    key = _parse_version(version)
    if key is None:
        return False
    return key[2] != _INFINITE or key[4] != _INFINITE


# ---------------------------------------------------------------------------
# State file (plain JSON, validated, atomically written)
# ---------------------------------------------------------------------------

def _state_file():
    override = os.environ.get(ENV_STATE_FILE, "").strip()
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), "feloopy", "Caches", "API", "update_check.json")


def _empty_state():
    return {
        "schema": STATE_SCHEMA,
        "next_check": 0.0,
        "last_check": 0.0,
        "latest": None,
        "notified": None,
        "failures": 0,
    }


def _load_state():
    state = _empty_state()
    try:
        # utf-8-sig: tolerate a BOM if the file was ever rewritten by a tool
        with open(_state_file(), "r", encoding="utf-8-sig") as handle:
            raw = json.load(handle)
    except Exception:
        return state
    if not isinstance(raw, dict):
        return state

    for key in ("next_check", "last_check"):
        value = raw.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            state[key] = float(value)

    for key in ("latest", "notified"):
        value = raw.get(key)
        if isinstance(value, str) and _VERSION_PATTERN.fullmatch(value):
            state[key] = value

    failures = raw.get("failures")
    if isinstance(failures, int) and not isinstance(failures, bool) and 0 <= failures <= 1000:
        state["failures"] = failures
    return state


def _save_state(state):
    path = _state_file()
    tmp_path = "{}.{}.tmp".format(path, os.getpid())
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, separators=(",", ":"))
        os.replace(tmp_path, path)          # atomic on POSIX and Windows
        return True
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False


def _acquire_lock():
    """Claim the cross-process check slot; returns the lock path or ``None``."""
    path = _state_file() + ".lock"
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    except Exception:
        return None

    for _attempt in (0, 1):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(descriptor, str(os.getpid()).encode("ascii", "ignore"))
            finally:
                os.close(descriptor)
            return path
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(path) <= LOCK_STALE_SECONDS:
                    return None              # a live process is checking now
                os.remove(path)              # stale lock: break it once
            except Exception:
                return None
        except Exception:
            return None
    return None


def _release_lock(path):
    try:
        if path:
            os.remove(path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------

def _truthy(name):
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


def _flag(name):
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return None


def _interval_seconds():
    raw = os.environ.get(ENV_INTERVAL)
    days = DEFAULT_INTERVAL_DAYS
    if raw:
        try:
            days = float(raw)
        except (TypeError, ValueError):
            days = DEFAULT_INTERVAL_DAYS
    if days != days or days in (float("inf"), float("-inf")):   # NaN/inf guard
        days = DEFAULT_INTERVAL_DAYS
    days = min(max(days, MIN_INTERVAL_SECONDS / 86400.0), MAX_INTERVAL_DAYS)
    return days * 86400.0


def _jitter_seconds():
    return random.uniform(-JITTER_SECONDS, JITTER_SECONDS)


def _backoff_seconds(failures):
    failures = max(1, min(int(failures), 30))
    days = min(float(2 ** (failures - 1)), float(MAX_BACKOFF_DAYS))
    return days * 86400.0


def _is_source_checkout():
    """True when running from a git checkout or an unpacked source tree."""
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(os.path.join(os.path.dirname(package_root), ".git")):
        return True
    try:
        from importlib import metadata
        metadata.version("feloopy")
    except Exception:
        return True                          # not installed as a distribution
    return False


def _auto_check_enabled():
    if _truthy(ENV_DISABLE):
        return False
    override = _flag(ENV_ENABLE)
    if override is not None:
        return override
    if getattr(sys.flags, "isolated", 0):
        return False
    if _truthy("CI"):
        return False
    if sys.stderr is None or getattr(sys.stderr, "closed", False):
        return False
    if _is_source_checkout():
        return False
    return True


# ---------------------------------------------------------------------------
# Network access
# ---------------------------------------------------------------------------

class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: the endpoint is fixed and never changes host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(newurl, code, msg, headers, fp)


def _fetch_latest_version(timeout=None):
    """Return the latest released version string from PyPI.

    Raises on any problem (HTTP error, timeout, malformed payload); callers
    translate that into backoff.  Never raises ``KeyboardInterrupt`` paths --
    only ``Exception`` subclasses produced by I/O and parsing.
    """
    timeout = TIMEOUT_SECONDS if timeout is None else float(timeout)
    request = urllib.request.Request(
        PYPI_URL,
        headers={
            "User-Agent": "feloopy/{} (+{})".format(__version__, UPGRADE_URL),
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Connection": "close",
        },
        method="GET",
    )
    opener = urllib.request.build_opener(_NoRedirectHandler)
    response = opener.open(request, timeout=timeout)
    try:
        status = getattr(response, "status", None) or response.getcode()
        if status != 200:
            raise urllib.error.HTTPError(PYPI_URL, status, "unexpected status", {}, None)
        payload = response.read(MAX_RESPONSE_BYTES + 1)
    finally:
        try:
            response.close()
        except Exception:
            pass

    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError("version payload exceeds {} bytes".format(MAX_RESPONSE_BYTES))

    data = json.loads(payload)
    info = data.get("info") if isinstance(data, dict) else None
    version = info.get("version") if isinstance(info, dict) else None
    if not isinstance(version, str) or not _VERSION_PATTERN.fullmatch(version):
        raise ValueError("version payload did not contain a usable version")
    if _parse_version(version) is None:
        raise ValueError("version payload contained an unparsable version")
    return version


# ---------------------------------------------------------------------------
# Notification output
# ---------------------------------------------------------------------------

def _color_supported(stream):
    try:
        if not stream.isatty():
            return False
    except Exception:
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    return True


def _announce(latest):
    """Print the one-line notice; returns True when it was actually written."""
    message = (
        "FelooPy {} is available (you have {}) -> "
        "python -m pip install --upgrade feloopy"
    ).format(latest, __version__)
    try:
        stream = sys.stderr
        if stream is None or getattr(stream, "closed", False):
            return False
        if _color_supported(stream):
            message = "\033[36;2m{}\033[0m".format(message)
        stream.write(message + "\n")
        stream.flush()
        return True
    except Exception:
        return False


def _print_status(result):
    try:
        lines = [
            "FelooPy update check",
            "  installed: {}".format(result["current"]),
            "  on PyPI:   {}".format(result["latest"] or "unknown"),
            "  update:    {}".format("yes" if result["update_available"] else "no"),
        ]
        if result.get("checked"):
            lines.append("  checked:   just now")
        elif result.get("next_check"):
            remaining = max(0.0, result["next_check"] - time.time())
            lines.append("  next check: in {:.1f} days".format(remaining / 86400.0))
        if result.get("error"):
            lines.append("  error:     {}".format(result["error"]))
        if result["update_available"]:
            lines.append("  upgrade:   python -m pip install --upgrade feloopy")
            lines.append("  releases:  {}".format(RELEASES_URL))
        sys.stderr.write("\n".join(lines) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _safe_error(exc):
    return "{}: {}".format(type(exc).__name__, exc)[:200]


# ---------------------------------------------------------------------------
# The check itself
# ---------------------------------------------------------------------------

def _perform_check(state, force, timeout):
    """Fetch the latest version, updating *state*.  Returns (state, checked, error)."""
    now = time.time()
    lock = _acquire_lock()
    if lock is None and not force:
        return state, False, None            # another process is checking

    try:
        if not force:
            # Write-ahead claim: even if we crash mid-request, no other process
            # will re-hit PyPI until the interval (or the backoff) elapses.
            state["next_check"] = now + _interval_seconds() + _jitter_seconds()
            if not _save_state(state):
                # Fail closed: without a persisted claim, every later run would
                # look "due" and re-query PyPI.  One silent skipped check is
                # always better than a request per import.
                return state, False, "update-check state file is not writable"

        try:
            latest = _fetch_latest_version(timeout=timeout)
        except Exception as exc:
            state["failures"] = min(int(state.get("failures", 0)) + 1, 1000)
            state["next_check"] = now + _backoff_seconds(state["failures"])
            _save_state(state)
            return state, True, _safe_error(exc)

        state["failures"] = 0
        state["last_check"] = now
        state["latest"] = latest
        state["next_check"] = now + _interval_seconds() + _jitter_seconds()
        _save_state(state)
        return state, True, None
    finally:
        if lock is not None:
            _release_lock(lock)


def check_update(force=False, verbose=True, notify=True, timeout=None):
    """Check PyPI for a newer FelooPy release.

    Never raises.  Without ``force`` the check is throttled by the on-disk
    schedule (default: one request per machine per week) and skips the network
    entirely when the cached result is still fresh -- in that case any known
    update is still announced if it has not been shown yet.

    Args:
        force: Bypass the schedule and query PyPI now (user-initiated, e.g.
            ``feloopy update-check``).  Concurrency lock is still honoured.
        verbose: Print a short status report to stderr.
        notify: Print the one-line upgrade notice when an update exists and it
            has not been shown for this version yet.
        timeout: Socket timeout in seconds (defaults to 3).

    Returns:
        dict with ``current``, ``latest``, ``update_available``, ``checked``,
        ``next_check``, ``notified`` and ``error`` keys.
    """
    result = {
        "current": __version__,
        "latest": None,
        "update_available": False,
        "checked": False,
        "notified": False,
        "next_check": None,
        "error": None,
    }
    try:
        state = _load_state()
        now = time.time()
        due = now >= float(state.get("next_check", 0.0))

        if force or due:
            state, checked, error = _perform_check(state, force=force, timeout=timeout)
            result["checked"] = checked
            result["error"] = error

        result["latest"] = state.get("latest")
        result["next_check"] = state.get("next_check")
        result["update_available"] = _is_newer(result["latest"], __version__)

        if notify and result["update_available"] and state.get("notified") != result["latest"]:
            if _announce(result["latest"]):
                state["notified"] = result["latest"]
                _save_state(state)
                result["notified"] = True

        if verbose:
            _print_status(result)
    except Exception as exc:
        result["error"] = result["error"] or _safe_error(exc)
        if verbose:
            try:
                _print_status(result)
            except Exception:
                pass
    return result


def _background_check():
    try:
        check_update(force=False, verbose=False, notify=True)
    except Exception:
        pass


def schedule_update_check():
    """Kick off the automatic, throttled check in a daemon thread.

    Called from ``feloopy/__init__``.  Returns True when a thread was started.
    Adding the thread costs microseconds; the import is never blocked and the
    thread never outlives the interpreter (it is a daemon).
    """
    global _scheduled
    try:
        if not _auto_check_enabled():
            return False
        with _schedule_lock:
            if _scheduled:
                return False
            _scheduled = True
        thread = threading.Thread(
            target=_background_check,
            name="feloopy-update-check",
            daemon=True,
        )
        thread.start()
        return True
    except Exception:
        return False


def reset_schedule_cache():
    """Testing helper: allow ``schedule_update_check`` to run again."""
    global _scheduled
    with _schedule_lock:
        _scheduled = False
