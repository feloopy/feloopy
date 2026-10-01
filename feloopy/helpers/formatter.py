# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from datetime import datetime
from contextlib import contextmanager
import threading
import time
import os
import sys

def _enable_vt():
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass

_enable_vt()

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def clear_console():
    """Clear the terminal screen.
    """
    try:
        if not sys.stdout.isatty():
            return
    except Exception:
        return
    sys.stdout.write("\033[2J\033[3J\033[H")
    sys.stdout.flush()


def _make_console():
    from rich.console import Console
    return Console()


_console = None
def _get_console():
    global _console
    if _console is None:
        _console = _make_console()
    return _console
_spinner_thread = None
_spinner_running = threading.Event()
_is_notebook = False
_start_time = None
_show_elapsed = False
_pending_message = None
_progress_message = None
_progress_bypass_fd = None
_progress_color = "\033[36m"
_progress_lock = threading.RLock()
_progress_suppressed = 0
_notebook_display_handle = None
_notebook_completed_html = []  


def _detect_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and "IPKernelApp" in ip.config:
            return True
    except Exception:
        pass
    return False


def _get_tqdm():
    if _detect_notebook():
        from tqdm.notebook import tqdm as _tqdm
    else:
        from tqdm import tqdm as _tqdm
    return _tqdm


def _format_elapsed(delta):
    total_seconds = int(delta.total_seconds())
    periods = [
        ('week', 60 * 60 * 24 * 7),
        ('day', 60 * 60 * 24),
        ('hour', 60 * 60),
        ('minute', 60),
        ('second', 1),
    ]
    strings = []
    for name, count in periods:
        value, total_seconds = divmod(total_seconds, count)
        if value:
            strings.append(f"{value} {name}{'s' if value > 1 else ''}")
    return ", ".join(strings) if strings else "0 seconds"


def _rich_print(message):
    if _is_notebook:
        from rich.jupyter import print as jupyter_print
        jupyter_print(message)
    else:
        _get_console().print(message)


def run_with_progress(func, show_log, *args, **kwargs):
    stop_event = threading.Event()
    _tqdm = _get_tqdm()

    def show_progress():
        with _tqdm(total=0, unit="s", bar_format="{desc}") as pbar:
            while not stop_event.is_set():
                pbar.set_description("Processing...")
                pbar.update()
                time.sleep(0.5)

    if show_log:
        progress_thread = threading.Thread(target=show_progress)
        progress_thread.start()

    try:
        func(*args, **kwargs)
    finally:
        if show_log:
            stop_event.set()
            progress_thread.join()


def progress_bar(iterable, unit="iter", description="Progress", remain=False):
    _tqdm = _get_tqdm()
    leave = False if remain == False else True
    return _tqdm(iterable, desc=description, unit=unit, ncols=82, leave=leave)


_CYAN = "\033[36m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_BOLD_CYAN = "\033[1;36m"
_BOLD_GREEN = "\033[1;32m"
_BOLD_RED = "\033[1;31m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_EL = "\033[K"

_SPIN_FRAMES = ["✳", "✴", "✶", "✱", "✦"]
_CHECK = "✓"
_CROSS = "✗"

_use_colorama = False
try:
    import colorama as _colorama
    _colorama.init(autoreset=True, strip=False)
    _use_colorama = True
    for _stream_name in ('stdout', 'stderr'):
        _stream = getattr(sys, _stream_name, None)
        if _stream is not None and type(_stream).__name__ == 'StreamWrapper':
            _orig_write = _stream.write
            def _write_returning_int(text, _orig=_orig_write):
                _orig(text)
                return len(text)
            _stream.write = _write_returning_int
except Exception:
    pass


def set_progress_bypass(fd):
    """Write the spinner directly to ``fd`` instead of stdout.

    """
    global _progress_bypass_fd
    _progress_bypass_fd = fd


def _write_bypass(fd, text):
    """Write ``text`` to a saved descriptor without going through the
    redirected stdout.

    """
    if sys.platform == "win32":
        try:
            import ctypes
            import msvcrt
            _handle = msvcrt.get_osfhandle(fd)
            _nchars = len(text.encode("utf-16-le")) // 2
            _buf = ctypes.create_unicode_buffer(_nchars + 1)
            _buf.value = text
            _written = ctypes.c_ulong(0)
            if ctypes.windll.kernel32.WriteConsoleW(
                    _handle, _buf, _nchars, ctypes.byref(_written), None):
                return
        except Exception:
            pass
    try:
        os.write(fd, text.encode("utf-8", errors="replace"))
    except Exception:
        pass


def _cwrite(text):
    _fd = _progress_bypass_fd
    if _fd is not None:
        _write_bypass(_fd, text)
        return
    sys.stdout.write(text)
    sys.stdout.flush()


def _notebook_html(text):
    """Render a rich Text to the HTML rich uses for Jupyter output cells.
    """
    from rich.console import Console
    from rich.jupyter import _render_segments

    console = Console()
    segments = list(console.render(text, console.options))
    while segments:
        last = segments[-1]
        if not last.text:
            segments.pop()
            continue
        trimmed = last.text.rstrip("\n")
        if trimmed != last.text:
            if trimmed:
                segments[-1] = last._replace(text=trimmed)
            else:
                segments.pop()
        break
    return _render_segments(segments).rstrip("\n")


def _notebook_status_html(symbol, message, elapsed_str, style):
    """Render a final ``✓ Done`` status line as notebook HTML.
    """
    from rich.text import Text

    text = Text()
    text.append(f"{symbol} {message}", style=style)
    text.append(elapsed_str, style="dim")
    return _notebook_html(text)


@contextmanager
def suppress_progress():
    """Disable the process-wide progress spinner inside this block.

    """
    global _progress_suppressed
    with _progress_lock:
        _progress_suppressed += 1
    try:
        yield
    finally:
        with _progress_lock:
            _progress_suppressed -= 1


@contextmanager
def suppress_output():
    """Silence stdout/stderr inside this block, including raw file
    descriptor output (solver/subprocess logs).

    """

    _devnull = open(os.devnull, 'w', encoding='utf-8', errors='replace')
    from contextlib import redirect_stdout, redirect_stderr
    _stdout_fd = _stderr_fd = None
    _saved_stdout = _saved_stderr = None
    try:
        try:
            _stdout_fd = sys.stdout.fileno()
            _stderr_fd = sys.stderr.fileno()
            _saved_stdout = os.dup(_stdout_fd)
            _saved_stderr = os.dup(_stderr_fd)
        except Exception:
            # no real file descriptor (notebook / StringIO capture):
            # Python-level redirection only, never raise here
            if _saved_stdout is not None:
                try:
                    os.close(_saved_stdout)
                except Exception:
                    pass
            _saved_stdout = _saved_stderr = None
            _stdout_fd = _stderr_fd = None
        if _saved_stdout is not None:
            os.dup2(_devnull.fileno(), _stdout_fd)
            os.dup2(_devnull.fileno(), _stderr_fd)
            set_progress_bypass(_saved_stdout)
        with redirect_stdout(_devnull), redirect_stderr(_devnull):
            yield
    finally:
        if _saved_stdout is not None:
            try:
                os.dup2(_saved_stdout, _stdout_fd)
                os.dup2(_saved_stderr, _stderr_fd)
            except Exception:
                pass
            # never leave the spinner pointed at a dead descriptor,
            # even if the descriptor restore failed
            set_progress_bypass(None)
            try:
                os.close(_saved_stdout)
                os.close(_saved_stderr)
            except Exception:
                pass
        _devnull.close()


def start_progress(message="Processing...", spinner="dots", show_elapsed=False, color=None):
    if _progress_suppressed:
        return
    with _progress_lock:
        _start_progress_locked(message, spinner, show_elapsed, color)


def update_progress(message):
    """Live-update the message of a running spinner.

    """
    global _progress_message
    if _progress_suppressed or not _spinner_running.is_set():
        return
    with _progress_lock:
        _progress_message = str(message)
        if not _is_notebook and _start_time is not None:
            elapsed = (datetime.now() - _start_time).total_seconds()
            elapsed_str = f" {elapsed:.1f}s" if elapsed >= 1 else ""
            _cwrite(f"\r{_progress_color}{_SPIN_FRAMES[0]} "
                    f"{_progress_message}{_DIM}{elapsed_str}{_RESET}{_EL}")



def _start_progress_locked(message, spinner, show_elapsed, color):
    global _spinner_running, _spinner_thread, _is_notebook, _start_time, _show_elapsed
    global _progress_message, _progress_color, _notebook_display_handle, _notebook_completed_html

    if _spinner_thread is not None and _spinner_thread.is_alive():
        _spinner_running.clear()
        if _spinner_thread.ident is not None:
            _spinner_thread.join(timeout=1.0)
        _spinner_thread = None

    _is_notebook = _detect_notebook()
    _start_time = datetime.now()
    _show_elapsed = show_elapsed
    _progress_message = message

    _color_code = {"cyan": _CYAN, "blue": "\033[34m", "green": _GREEN, "yellow": "\033[33m", "red": _RED}.get(color, _CYAN)
    _progress_color = _color_code

    if _is_notebook:
        from IPython.display import display, HTML
        
        _frame = [0]
        color_map = {"cyan": "#00bcd4", "blue": "#2196f3", "green": "#4caf50", 
                     "yellow": "#ff9800", "red": "#f44336"}
        css_color = color_map.get(color, "#00bcd4")
        
        def spinner_task():
            global _notebook_display_handle
            while _spinner_running.is_set():
                try:
                    ch = _SPIN_FRAMES[_frame[0] % len(_SPIN_FRAMES)]
                    elapsed = (datetime.now() - _start_time).total_seconds()
                    elapsed_str = f" {elapsed:.1f}s" if elapsed >= 1 else ""
                    
                    spinner_html = (f'<div style="font-family:monospace;margin:2px 0">'
                                   f'<span style="color:{css_color};font-weight:bold">{ch} {message}</span>'
                                   f'<span style="color:#888">{elapsed_str}</span></div>')
                    
                    full_html = "".join(_notebook_completed_html) + spinner_html
                    
                    if _notebook_display_handle is None:
                        _notebook_display_handle = display(HTML(full_html), display_id=True)
                    else:
                        _notebook_display_handle.update(HTML(full_html))
                    
                    _frame[0] += 1
                except Exception:
                    pass
                time.sleep(0.08)
        
        _spinner_running.set()
        _spinner_thread = threading.Thread(target=spinner_task, daemon=True)
        _spinner_thread.start()
    
    else:

        _frame = [0]

        def _spinner_tick():
            while _spinner_running.is_set():
                try:
                    ch = _SPIN_FRAMES[_frame[0] % len(_SPIN_FRAMES)]
                    elapsed = (datetime.now() - _start_time).total_seconds()
                    elapsed_str = f" {elapsed:.1f}s" if elapsed >= 1 else ""
                    line = f"\r{_color_code}{ch} {_progress_message}{_DIM}{elapsed_str}{_RESET}  "
                    _cwrite(line)
                    _frame[0] += 1
                except Exception:
                    pass
                time.sleep(0.08)

        _spinner_running.set()
        _spinner_thread = threading.Thread(target=_spinner_tick, daemon=True)
        _spinner_thread.start()


def end_progress(success_message="Done!", failure_message=None, success=True, show_elapsed=None):
    global _spinner_running, _spinner_thread, _notebook_display_handle, _notebook_completed_html

    if _progress_suppressed:
        return
    with _progress_lock:
        _spinner_running.clear()
        thread = _spinner_thread
        _spinner_thread = None
    if thread is not None and thread.ident is not None:
        thread.join()

    final_show = _show_elapsed if show_elapsed is None else show_elapsed
    elapsed_str = ""
    if final_show and _start_time:
        elapsed_str = f" ({_format_elapsed(datetime.now() - _start_time)})"

    if _is_notebook:
        from IPython.display import HTML
        
        symbol = _CHECK if success else _CROSS
        css_color = "#4caf50" if success else "#f44336"
        label = success_message if (success and success_message) else (failure_message or "")
        
        if label:
            # Add this completion to the accumulated list
            completed_html = (f'<div style="font-family:monospace;margin:2px 0">'
                            f'<span style="color:{css_color};font-weight:bold">{symbol} {label}</span>'
                            f'<span style="color:#888">{elapsed_str}</span></div>')
            _notebook_completed_html.append(completed_html)
            
            # Update the display with all completed lines (no spinner now)
            full_html = "".join(_notebook_completed_html)
            
            if _notebook_display_handle is not None:
                try:
                    _notebook_display_handle.update(HTML(full_html))
                    # Keep the handle alive for next operation
                except Exception:
                    from IPython.display import display
                    _notebook_display_handle = display(HTML(full_html), display_id=True)
            else:
                from IPython.display import display
                _notebook_display_handle = display(HTML(full_html), display_id=True)
        return

    # Terminal mode
    _pad = " " * 40
    if success and success_message:
        _cwrite(f"\r{_BOLD_GREEN}{_CHECK} {success_message}{_DIM}{elapsed_str}{_RESET}{_pad}\n")
    elif failure_message:
        _cwrite(f"\r{_BOLD_RED}{_CROSS} {failure_message}{_DIM}{elapsed_str}{_RESET}{_pad}\n")
