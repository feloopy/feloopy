# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import os
import glob
import platform
import subprocess
import tempfile
import time

_TEMP_PREFIX = 'feloopy_picat_'
_TEMP_SUFFIX = '.pi'
_orphans_cleaned = False


def _find_picat_binary():
    """Find the Picat binary on the system."""
    system = platform.system().lower()

    if system == 'windows':
        binary_name = 'picat.exe'
    else:
        binary_name = 'picat'

    search_paths = [
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Interpreters', 'Picat'),
        os.path.expanduser(os.path.join('~', 'feloopy', 'Interpreters', 'Picat')),
        os.path.expanduser(os.path.join('~', '.feloopy', 'Interpreters', 'Picat')),
    ]

    for base_path in search_paths:
        base_path = os.path.normpath(base_path)
        if not os.path.isdir(base_path):
            continue
        for entry in os.listdir(base_path):
            entry_path = os.path.join(base_path, entry)
            if os.path.isdir(entry_path):
                candidate = os.path.join(entry_path, binary_name)
                if os.path.isfile(candidate):
                    return candidate
        candidate = os.path.join(base_path, binary_name)
        if os.path.isfile(candidate):
            return candidate

    try:
        result = subprocess.run(
            ['where', binary_name] if system == 'windows' else ['which', binary_name],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')[0].strip()
    except Exception:
        pass

    return None


def get_picat_binary():
    """Get the Picat binary path, raising an error if not found."""
    binary = _find_picat_binary()
    if binary is None:
        raise FileNotFoundError(
            "Picat binary not found. Please download Picat from http://picat-lang.org/download.html "
            "and place it in one of these locations:\n"
            "  - <feloopy_package>/Interpreters/Picat/<version>/\n"
            "  - ~/feloopy/Interpreters/Picat/<version>/\n"
            "  - ~/.feloopy/Interpreters/Picat/<version>/\n"
            "Or ensure 'picat' is on your system PATH.\n"
            "Alternatively, use the FelooPy IDE to set up Picat."
        )
    return binary


def _remove_temp_file(path):
    """Remove a temp file with a rename-then-delete fallback for Windows locking."""
    try:
        os.remove(path)
        return
    except OSError:
        pass
    try:
        pending = path + '._pending_delete'
        os.rename(path, pending)
        os.remove(pending)
    except OSError:
        pass


def cleanup_temp_files():
    """Remove orphaned feloopy picat temp files from the system temp directory."""
    temp_dir = tempfile.gettempdir()
    pattern = os.path.join(temp_dir, f'{_TEMP_PREFIX}*{_TEMP_SUFFIX}')
    for path in glob.glob(pattern):
        _remove_temp_file(path)


def run_picat(source_code, time_limit=None, picat_binary=None):
    """Execute Picat source code and return (stdout, stderr, exit_code, elapsed_time)."""
    global _orphans_cleaned
    if not _orphans_cleaned:
        cleanup_temp_files()
        _orphans_cleaned = True

    if picat_binary is None:
        picat_binary = get_picat_binary()

    if not os.path.isfile(picat_binary):
        raise FileNotFoundError(
            f"Picat binary not found at: {picat_binary}\n"
            "Please verify the installation path or reinstall Picat."
        )

    temp_fd, temp_path = tempfile.mkstemp(suffix=_TEMP_SUFFIX, prefix=_TEMP_PREFIX)
    proc = None
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            f.write(source_code)

        cmd = [picat_binary, temp_path]
        start_time = time.time()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = proc.communicate(timeout=time_limit)
            elapsed = time.time() - start_time
            return stdout, stderr, proc.returncode, elapsed
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            if proc is not None:
                proc.kill()
                proc.communicate()
            return '', 'Picat execution timed out', 1, elapsed
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.communicate()
        _remove_temp_file(temp_path)
