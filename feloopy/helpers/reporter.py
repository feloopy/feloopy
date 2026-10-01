# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from __future__ import annotations

import os
import sys
import io
import time as _time
import base64
import numpy as np
from numbers import Number
from typing import Any, Sequence, Optional
from ._lazy import pl


def _is_modern_terminal():
    env = os.environ
    if env.get('WT_SESSION'):
        return True
    term_program = env.get('TERM_PROGRAM', '').lower()
    if term_program in ('vscode', 'iterm.app', 'iterm2', 'apple_terminal', 'alacritty', 'kitty', 'wezterm', 'mintty', 'conemu', 'cmder'):
        return True
    term = env.get('TERM', '').lower()
    if 'xterm' in term or 'screen' in term or 'tmux' in term or 'linux' in term:
        return True
    if env.get('COLORTERM'):
        return True
    if env.get('TERM_PROGRAM_VERSION'):
        return True
    return False


def _copy_to_clipboard_osc52(text):
    try:
        encoded = base64.b64encode(text.encode('utf-8')).decode('ascii')
        sequence = f"\033]52;c;{encoded}\033\\"
        sys.stdout.write(sequence)
        sys.stdout.flush()
        return True
    except Exception:
        return False


def _copy_to_clipboard_fallback(text):
    import subprocess
    import platform
    try:
        system = platform.system()
        if system == "Windows":
            process = subprocess.Popen(['clip'], stdin=subprocess.PIPE)
            process.communicate(input=text.encode('utf-16-le'))
            return True
        elif system == "Darwin":
            process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
            process.communicate(input=text.encode('utf-8'))
            return True
        else:
            for cmd in [['xclip', '-selection', 'clipboard'], ['xsel', '--clipboard', '--input']]:
                try:
                    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                    process.communicate(input=text.encode('utf-8'))
                    return True
                except FileNotFoundError:
                    continue
    except Exception:
        pass
    return False


def _read_mouse_event(timeout=0.5):
    import select
    import platform
    if platform.system() == "Windows":
        import msvcrt
        import time
        start = time.time()
        while time.time() - start < timeout:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch == '\x1b':
                    seq = ch
                    while True:
                        if msvcrt.kbhit():
                            ch = msvcrt.getwch()
                            seq += ch
                            if ch == 'M' or ch == 'm':
                                break
                        else:
                            break
                    if len(seq) >= 6 and seq[-1] in ('M', 'm'):
                        cb = ord(seq[3])
                        cx = ord(seq[4]) - 32
                        cy = ord(seq[5]) - 32
                        pressed = seq[-1] == 'M'
                        return (cx, cy, pressed)
                elif ch in ('\r', '\n', ' '):
                    return None
            time.sleep(0.01)
        return None
    else:
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            data = sys.stdin.read(1)
            if data == '\x1b':
                buf = data
                rlist2, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist2:
                    buf += sys.stdin.read(6)
                if len(buf) >= 6 and buf[-1] in ('M', 'm'):
                    cb = ord(buf[3])
                    cx = ord(buf[4]) - 32
                    cy = ord(buf[5]) - 32
                    pressed = buf[-1] == 'M'
                    return (cx, cy, pressed)
        return None


# Top-right marker of a per-parameter sensitivity box: which way to move that
# parameter to improve the objective.  Key = the glyph stored in
# _impact['direction_effect'], value = the short word rendered beside it so the
# marker stays readable without consulting the legend.
DIRECTION_HINTS = {
    "↑": "raise",
    "↓": "lower",
    "─": "none",
    "↕": "mixed",
}


class report:

    def __init__(
        self,
        topleft="Result",
        topcenter="",
        topright="",
        bottomleft="",
        bottomcenter="",
        bottomright="",
        width=78,
        style=1,
        sci_thresh=8,
    ):

        self.styles = {
            0: ["╭", "─", "╮", "╰", "─", "╯", "├", "┤", "┬", "┴", "│", "─"],
            1: ["┌", "─", "┐", "└", "─", "┘", "╞", "╡", "┬", "┴", "│", "═"],
            2: ["+", "-", "+", "+", "-", "+", "|", "|", "-", "-", "|", "="],
            3: ["╔", "═", "╗", "╚", "═", "╝", "╠", "╣", "╦", "╩", "║", "═"],
            4: ["┏", "━", "┓", "┗", "━", "┛", "┣", "┫", "┳", "┻", "┃", "━"],
            5: ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#"],
            6: ["/", "-", "\\", "\\", "-", "/", "|", "|", "^", "v", "|", "-"],
            7: ["*", "*", "*", "*", "*", "*", "*", "*", "*", "*", "*", "*"],
            8: ["=", "=", "=", "=", "=", "=", "=", "=", "=", "=", "=", "="],
            9: ["▛", "▀", "▜", "▙", "▄", "▟", "▌", "▐", "▀", "▄", "█", "█"],
            10: ["█", "▀", "█", "█", "▄", "█", "█", "█", "▀", "▄", "█", "█"],
            11: ["▓", "▓", "▓", "▓", "▓", "▓", "▓", "▓", "▓", "▓", "▓", "▓"],
            12: ["▘", "▀", "▝", "▖", "▄", "▗", "▌", "▐", "▀", "▄", "▌", "▔"],
        }

        self._topleft = self.styles[style][0]
        self._topright = self.styles[style][2]
        self._topbar = self.styles[style][1]
        self._bottomleft = self.styles[style][3]
        self._bottombar = self.styles[style][4]
        self._bottomright = self.styles[style][5]
        self._middleleft = self.styles[style][6]
        self._middleright = self.styles[style][7]
        self._middletop = self.styles[style][8]
        self._middlebottom = self.styles[style][9]
        self._border = self.styles[style][10]
        self._middlebar = self.styles[style][11]
        self.current_style= style

        self.width = width
        self.topleft = topleft
        self.topcenter = topcenter
        self.topright = topright
        self.bottomleft = bottomleft
        self.bottomcenter = bottomcenter
        self.bottomright = bottomright

        self.start_top = self._topleft + self._topbar
        self.end_top = self._topbar + self._topright
        self.start_bottom = self._bottomleft + self._bottombar
        self.end_bottom = self._bottombar + self._bottomright
        self.start_middle = self._middleleft + self._middlebar
        self.end_middle = self._middlebar + self._middleright
        self.sci_thresh = sci_thresh

        self._buffered = False
        self._lines = []

    @staticmethod
    def _visible_len(text):
        """Return visible character count, ignoring ANSI escape codes."""
        import re
        return len(re.sub(r'\x1b\[[0-9;]*m', '', str(text)))

    def _clip(self, text, max_width):
        if max_width <= 0:
            return ""
        text = str(text)
        if self._visible_len(text) <= max_width:
            return text
        if max_width <= 3:
            # clip by visible length
            import re
            clean = re.sub(r'\x1b\[[0-9;]*m', '', text)
            return clean[:max_width]
        # clip by visible length, then re-apply trailing reset
        import re
        clean = re.sub(r'\x1b\[[0-9;]*m', '', text)
        from colorama import Fore
        return clean[:max_width - 3] + "..." + Fore.RESET

    def _emit(self, text, line_type="raw"):
        if self._buffered:
            self._lines.append((line_type, text))
        else:
            print(text)

    def capture(self, func, *args, **kwargs):
        if not self._buffered:
            func(*args, **kwargs)
            return
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout
        for line in buf.getvalue().rstrip("\n").split("\n"):
            if line:
                self._lines.append(("raw", line))

    def render(self, copy_to_clipboard=False):
        if not self._buffered:
            return
        all_lines = []
        for _, text in self._merge_boxes():
            all_lines.append(text)
        self._lines = []
        self._buffered = False

        if copy_to_clipboard is None:
            copy_to_clipboard = _is_modern_terminal()

        for line in all_lines:
            print(line)

        if copy_to_clipboard:
            full_text = "\n".join(all_lines)
            btn = "\u2398 Copy to clipboard"
            btn_width = len(btn) + 4
            print(f"  \033[1;36m {btn} \033[0m", end="", flush=True)
            try:
                sys.stdout.write("\033[?1000h")
                sys.stdout.write("\033[?1006h")
                sys.stdout.flush()
                result = _read_mouse_event(timeout=2.0)
                sys.stdout.write("\033[?1000l")
                sys.stdout.write("\033[?1006l")
                sys.stdout.flush()
                if result:
                    cx, cy, pressed = result
                    print()
                    if pressed:
                        _copy_to_clipboard_fallback(full_text)
                        print(f"  \033[1;32m\u2713 Copied!\033[0m")
                    else:
                        print()
                else:
                    print()
            except Exception:
                print()

    def _merge_boxes(self):
        result = []
        lines = self._lines
        i = 0
        while i < len(lines):
            lt, text = lines[i]
            if lt == "bottom" and i + 1 < len(lines) and lines[i + 1][0] == "top":
                bottom_text = text
                prev_top = None
                for j in range(i - 1, -1, -1):
                    if lines[j][0] == "top":
                        prev_top = lines[j][1]
                        break
                i += 1
                next_lt, next_text = lines[i]
                tchar = self._topbar
                merged_content = next_text[2:-2]
                if "\u2534" in bottom_text:
                    bot_content = bottom_text[2:-2]
                    try:
                        sep_pos = bot_content.index("\u2534")
                        merged_content = merged_content[:sep_pos] + "\u2534" + merged_content[sep_pos + 1:]
                    except ValueError:
                        pass
                new_text = "\u251c" + tchar + merged_content + tchar + "\u2524"
                result.append((next_lt, new_text))
                i += 1
                continue
            result.append((lt, text))
            i += 1
        return result

    def top(self, left="", center="", right=""):

        left = self._clip(left, self.width - 4)
        right = self._clip(right, self.width - 4)
        center = self._clip(center, self.width - 4)

        len_left = self._visible_len(left)
        len_right = self._visible_len(right)
        len_center = self._visible_len(center)

        if len_right > 0:
            right_space = " "
        else:
            right_space = ""

        if len_left > 0:
            left_space = " "
        else:
            left_space = ""

        if len_center > 0:
            center_space = " "
        else:
            center_space = ""

        final_right_text = right_space + right + right_space
        final_left_text = left_space + left + left_space
        final_center_text = center_space + center + center_space

        remaining = (
            self.width
            - len_left
            - 2 * len(left_space)
            - len_right
            - 2 * len(right_space)
            - len_center
            - 2 * len(center_space)
            - 4
        )
        if remaining < 0:
            remaining = 0

        padding_left = self._topbar * (remaining // 2)
        padding_right = self._topbar * (remaining - len(padding_left))

        top_line = f"{self.start_top}{final_left_text}{padding_left}{final_center_text}{padding_right}{final_right_text}{self.end_top}"
        self._emit(top_line, "top")

    def bottom(self, left="", center="", right=""):

        left = self._clip(left, self.width - 4)
        right = self._clip(right, self.width - 4)
        center = self._clip(center, self.width - 4)

        len_left = self._visible_len(left)
        len_right = self._visible_len(right)
        len_center = self._visible_len(center)

        if len_right > 0:
            right_space = " "
        else:
            right_space = ""

        if len_left > 0:
            left_space = " "
        else:
            left_space = ""

        if len_center > 0:
            center_space = " "
        else:
            center_space = ""

        final_right_text = right_space + right + right_space
        final_left_text = left_space + left + left_space
        final_center_text = center_space + center + center_space

        remaining = self.width - len_left - 2 * len(left_space) - len_right - 2 * len(right_space) - len_center - 2 * len(center_space) - 4
        if remaining < 0:
            remaining = 0

        padding_left = self._bottombar * (remaining // 2)
        padding_right = self._bottombar * (remaining - len(padding_left))

        bottom_line = f"{self.start_bottom}{final_left_text}{padding_left}{final_center_text}{padding_right}{final_right_text}{self.end_bottom}"
        self._emit(bottom_line, "bottom")

    def empty(self):
        self._emit(self._border + (self.width - 2) * " " + self._border, "empty")

    def middle(self, left="", center="", right=""):

        left = self._clip(left, self.width - 4)
        right = self._clip(right, self.width - 4)
        center = self._clip(center, self.width - 4)

        len_left = self._visible_len(left)
        len_right = self._visible_len(right)
        len_center = self._visible_len(center)

        if len_right > 0:
            right_space = " "
        else:
            right_space = ""

        if len_left > 0:
            left_space = " "
        else:
            left_space = ""

        if len_center > 0:
            center_space = " "
        else:
            center_space = ""

        final_right_text = right_space + right + right_space
        final_left_text = left_space + left + left_space
        final_center_text = center_space + center + center_space

        remaining = self.width - len_left - 2 * len(left_space) - len_right - 2 * len(right_space) - len_center - 2 * len(center_space) - 4
        if remaining < 0:
            remaining = 0

        padding_left = self._middlebar * (remaining // 2)
        padding_right = self._middlebar * (remaining - len(padding_left))

        middle_line = f"{self.start_middle}{final_left_text}{padding_left}{final_center_text}{padding_right}{final_right_text}{self.end_middle}"
        self._emit(middle_line, "middle")


    def row(self, left="", center="", right="", fill_char=False):
        left = self._clip(left, self.width - 4)
        right = self._clip(right, self.width - 4)
        center = self._clip(center, self.width - 4)

        len_left = self._visible_len(left)
        len_right = self._visible_len(right)
        len_center = self._visible_len(center)

        left_space = " " if len_left > 0 else ""
        right_space = " " if len_right > 0 else ""
        center_space = " " if len_center > 0 else ""

        final_left_text = left_space + left + left_space
        final_right_text = right_space + right + right_space
        final_center_text = center_space + center + center_space

        used_width = (
            self._visible_len(final_left_text) +
            self._visible_len(final_center_text) +
            self._visible_len(final_right_text)
        )

        remaining = self.width - used_width - 2  # -2 for borders

        if remaining < 0:
            remaining = 0

        if fill_char:
            padding_left = self._middlebar * (remaining // 2)
            padding_right = self._middlebar * (remaining - len(padding_left))
        else:
            padding_left = " " * (remaining // 2)
            padding_right = " " * (remaining - len(padding_left))

        row_line = f"{self._border}{final_left_text}{padding_left}{final_center_text}{padding_right}{final_right_text}{self._border}"
        self._emit(row_line, "row")

    def _row_raw(self, content):
        """Emit a raw row without clipping or centering. Content padded with 1 space on left, 1 on right."""
        row_line = f"{self._border} {content} {self._border}"
        self._emit(row_line, "row")


    def columns(self, list_of_strings, label, max_space_between_elements=4):
        label = self._clip(label, self.width - 4)
        clipped = [self._clip(s, self.width - 4) for s in list_of_strings]
        max_string_length = max(len(s) for s in clipped)
        total_width = self.width

        label_width = len(label)
        remaining_width = (
            total_width
            - label_width
            - 4 * len(clipped)
            - len(clipped) * max_string_length
        )
        min_space_between_elements = 1

        if len(clipped) > 1:
            min_space_between_elements = max(remaining_width // (len(clipped) - 1), 0)

        space_between_elements = min(
            max_space_between_elements, min_space_between_elements
        )
        if space_between_elements < max_space_between_elements:
            extra_space = remaining_width - space_between_elements * (
                len(clipped) - 1
            )
            if extra_space < 0:
                extra_space = 0
        else:
            extra_space = 0

        rowstart = f"{self.start_middle} {label} {' ' * extra_space}"
        row = ""
        for string in clipped[:-1]:
            row += f"{' ' * (max_string_length - len(string))}{string}{' ' * space_between_elements}"
        row += f"{' ' * (max_string_length - len(clipped[-1]))}{clipped[-1]}"
        remaining = self.width - len(rowstart) - len(row) - 3
        if remaining < 0:
            remaining = 0
        row = rowstart + remaining * self._middlebar + " " + row
        self._emit(f"{row} {self._border}", "row")


    def clear_columns(
        self,
        list_of_strings: Sequence[str],
        label: str,
        max_space_between_elements: int = 4,
        extra_spaces: int = 0
    ) -> None:
        if extra_spaces < 0:
            raise ValueError("extra_spaces must be >= 0")
        clipped_label = self._clip(label, self.width - 4)
        clipped = [self._clip(s, self.width - 4) for s in list_of_strings]
        max_len = max(len(s) for s in clipped)
        label_part = f"{self._border} {clipped_label}"
        tail = " "+self._border
        used = len(label_part) + len(tail) + len(clipped) * max_len
        slots = max(len(clipped) - 1, 1)
        remaining_width = self.width - used - extra_spaces * slots
        if remaining_width < 0:
            remaining_width = 0
        base_space = remaining_width // slots
        space_between = min(base_space, max_space_between_elements)
        leftover = self.width - used - slots * (space_between + extra_spaces)
        if leftover < 0:
            leftover = 0
        left = label_part + " " * leftover
        gap = space_between + extra_spaces
        body = ""
        for i, s in enumerate(clipped):
            body += s.rjust(max_len)
            if i < len(clipped) - 1:
                body += " " * gap
        self._emit(f"{left}{body}{tail}", "row")

    def _format_value(self, val: Any, decimal_places: int) -> str:
        if isinstance(val, Number):
            fv = float(val)
            if fv == 0 or (abs(fv) < 0.5 * 10 ** (-decimal_places)):
                return "0"
            if fv.is_integer():
                return str(int(fv))
            s = f"{fv:.{decimal_places}f}"
            s = s.rstrip("0").rstrip(".")
            compact = s.replace("-", "").replace(".", "")
            if len(compact) > self.sci_thresh:
                sci = f"{fv:.{decimal_places}e}"
                mantissa, exp = sci.split("e")
                mantissa = mantissa.rstrip("0").rstrip(".")
                return f"{mantissa}e{exp}"
            return s
        return str(val)

    @staticmethod
    def _pack_entries(entries, inner):
        """Group ``(label, value)`` pairs into rows whose *aligned* width
        fits ``inner``.

        An entry only joins the current row when the row aligned to its own
        longest label/value still fits the box — padding every entry to a
        global maximum would push short values past the right border, where
        the line slice silently hides them.
        """
        def row_width(row):
            ml = max(len(lbl) for lbl, _ in row)
            mv = max(len(val) for _, val in row)
            # every entry in the row is padded to the row's maxima
            return len(row) * (ml + 3 + mv) + (len(row) - 1) * 4

        grid = []
        current = []
        for entry in entries:
            if not current or row_width(current + [entry]) <= inner:
                current.append(entry)
            else:
                grid.append(current)
                current = [entry]
        if current:
            grid.append(current)
        return grid

    @staticmethod
    def _render_entry_rows(rows):
        """Render packed rows: labels left-justified and values
        right-justified within each row."""
        lines = []
        for row in rows:
            ml = max(len(lbl) for lbl, _ in row)
            mv = max(len(val) for _, val in row)
            lines.append(
                "    ".join(f"{lbl.ljust(ml)} = {val.rjust(mv)}" for lbl, val in row)
            )
        return lines

    @staticmethod
    def _hang_indent(ln: str, inner: int) -> str:
        """Continuation indent that lines wrapped values up under their
        ``= value`` column (capped so continuations keep room to breathe)."""
        pos = ln.index(" = ") + 3 if " = " in ln else 4
        return " " * max(0, min(pos, inner - 20))

    def print_element(
        self,
        label: str,
        var: Any,
        additional_text: str = "",
        decimal_places: int = 4
    ) -> None:
        if isinstance(var, (np.generic, list, tuple)):
            # numpy scalars (np.int64, ...) and plain sequences (e.g. from
            # decoder results): display by their actual shape — 0-dim values
            # render as a bare label, sequences as indexed elements
            try:
                var = np.asarray(var)
            except Exception:
                pass  # ragged sequences fall through to the string form
        try:
            if isinstance(var, np.ndarray):
                self._print_numpy_element(label, var, decimal_places, additional_text=additional_text)
            elif isinstance(var, dict):
                self._print_dict_element(label, var, decimal_places, additional_text=additional_text)
            elif isinstance(var, (int, float)):
                inner = self.width - 4
                if var == 0:
                    content = f"{label} = all zeros"
                else:
                    content = f"{label} = {self._format_value(var, decimal_places)}"
                if additional_text:
                    content += additional_text
                self._emit(f"{self._border} {content.ljust(inner)} {self._border}", "row")
            else:
                raise ValueError("print_element only accepts np.ndarray, dict, int, or float")
        except Exception:
            # never drop an entry — one failure would otherwise abort the rest
            # of the decision section: fall back to its string form
            inner = self.width - 4
            content = f"{label} = {var}"
            if additional_text:
                content += additional_text
            self._emit_block(content, self._hang_indent(content, inner))

    def print_value(
        self,
        label: str,
        var: Any,
        additional_text: str = "",
        decimal_places: int = 4,
        font_mode: bool = False,
        hide_zero: bool = True,
        sign: str = "=",
    ) -> None:
        inner_width = self.width - 4  # account for borders and spaces

        if isinstance(var, np.ndarray):
            self._print_numpy_element(label, var, decimal_places)
        elif isinstance(var, dict):
            self._print_dict_element(label, var, decimal_places)
        elif isinstance(var, (int, float, str)):
            if isinstance(var, (int, float)) and hide_zero and abs(var) < 10**-decimal_places:
                return  # skip printing zeros
            formatted = f"{var:.{decimal_places}f}" if isinstance(var, float) else str(var)

            if font_mode:
                max_label_width = inner_width - len(formatted) - 3  # space for " : "
                label_str = label[:max(0, max_label_width)].ljust(max(0, max_label_width))
                line = f"{label_str} {sign} {formatted}"
            else:
                line = f"{label} {sign} {formatted}"

            self._emit(f"{self._border} {line[:inner_width].ljust(inner_width)} {self._border}", "row")
        else:
            raise ValueError("print_value only accepts np.ndarray, dict, int, float, or str")

        if additional_text:
            inner = self.width - 4
            self._emit(
                f"{self._border} "
                f"{additional_text[:inner].ljust(inner)} "
                f"{self._border}", "row"
            )


    def _print_numpy_element(
        self,
        label: str,
        arr: np.ndarray,
        decimal_places: int,
        additional_text: str = ""
    ) -> None:
        inner = self.width - 4

        # Split into 2-D matrices (leading two dims, trailing index pages) so
        # each matrix can be decoded as a route/assignment; 0-D/1-D arrays form
        # a single page without an annotation.
        if arr.ndim >= 2:
            trail = arr.shape[2:]
            keys = list(np.ndindex(*trail)) if trail else [()]
            pages = [(k, arr[(slice(None), slice(None)) + k]) for k in keys]
        else:
            pages = [((), arr)]

        emitted_any = False

        def emit_ln(ln: str, attach: str = ""):
            nonlocal emitted_any
            if attach and not emitted_any:
                ln = ln + attach
            # wrap (never trim) over-long values onto continuation lines
            self._emit_block(ln, self._hang_indent(ln, inner))
            emitted_any = True

        for trail_idx, page in pages:
            entries = []
            idx = [0] * page.ndim
            def collect(dim: int = 0, _page=page, _idx=idx, _trail=trail_idx):
                if dim < _page.ndim:
                    for i in range(_page.shape[dim]):
                        _idx[dim] = i
                        collect(dim + 1, _page, _idx, _trail)
                else:
                    val = _page[tuple(_idx)]
                    if abs(float(val)) >= 10**(-decimal_places):
                        full = tuple(_idx) + tuple(_trail)
                        index_str = "[" + ",".join(map(str, full)) + "]" if full else ""
                        entries.append((f"{label}{index_str}", self._format_value(val, decimal_places)))
            collect()
            if not entries:
                continue

            max_label = max(len(e[0]) for e in entries)
            if len(entries) == 1:
                emit_ln(f"{entries[0][0].ljust(max_label)} = {entries[0][1]}", additional_text)
            else:
                rows = self._pack_entries(entries, inner)
                for ln in self._render_entry_rows(rows):
                    emit_ln(ln, additional_text)
            # route/assignment annotations are a tensor-mode feature
            # (see render_tensor_blocks); element mode shows raw entries only

        if not emitted_any:
            content = f"{label} = all zeros"
            if additional_text:
                content += additional_text
            self._emit(f"{self._border} {content.ljust(inner)} {self._border}", "row")

    def _print_dict_element(
        self,
        label: str,
        d: dict,
        decimal_places: int,
        additional_text: str = ""
    ) -> None:
        inner = self.width - 4
        entries = []

        def key_part(k):
            ks = str(k).replace("(", "[").replace(")", "]")
            return ks if "[" in ks else f"[{ks}]"

        def add(name, raw):
            # scalars are zero-suppressed; anything non-numeric is shown as-is
            try:
                fv = float(raw)
            except (TypeError, ValueError):
                s = str(raw)
                entries.append((name, s if s else "''"))
                return
            if abs(fv) >= 10**(-decimal_places):
                entries.append((name, self._format_value(fv, decimal_places)))

        def walk(prefix, v):
            # scalar leaves: 0-d arrays, numpy scalars, python numbers/strings
            if isinstance(v, np.ndarray) and v.ndim == 0:
                add(prefix, v.item())
                return
            if not isinstance(v, (dict, list, tuple, np.ndarray)):
                add(prefix, v)
                return
            if isinstance(v, dict):
                # nested dict (e.g. per-picker mapping) extends the key path;
                # only the keys actually present are shown — dimensions need
                # not be complete
                for k2, v2 in v.items():
                    walk(prefix + key_part(k2), v2)
                return
            # sequences — stored as nested lists of plain numbers
            try:
                arr = np.asarray(v)
                idx_iter = np.ndindex(arr.shape) if arr.ndim else ((),)
            except Exception:
                entries.append((prefix, str(v)))  # ragged / exotic
                return
            for idx in idx_iter:
                suffix = "[" + ",".join(str(i) for i in idx) + "]" if idx else ""
                add(prefix + suffix, arr[idx])

        if not d:
            content = f"{label} = {{}}"
            if additional_text:
                content += additional_text
            self._emit(f"{self._border} {content[:inner].ljust(inner)} {self._border}", "row")
            return
        for k, v in d.items():
            walk(label + key_part(k), v)
        if not entries:
            content = f"{label} = all zeros"
            if additional_text:
                content += additional_text
            self._emit(f"{self._border} {content.ljust(inner)} {self._border}", "row")
            return
        max_label = max(len(e[0]) for e in entries)
        if len(entries) == 1:
            content = f"{entries[0][0].ljust(max_label)} = {entries[0][1]}"
            if additional_text:
                content += additional_text
            self._emit_block(content, self._hang_indent(content, inner))
            return
        for i, ln in enumerate(
            self._render_entry_rows(self._pack_entries(entries, inner))
        ):
            if i == 0 and additional_text:
                ln = ln + additional_text
            self._emit_block(ln, self._hang_indent(ln, inner))

    @staticmethod
    def _chunk_annotation(text: str, width: int) -> list:
        """Split an annotation like ``assignment: 0 -> [0, 2], 1->3`` into
        lines of at most ``width`` characters.

        Breaks only on top-level ``", "`` boundaries — commas inside ``[...]``
        groups stay with their group.
        """
        # split on ", " that are not inside brackets
        parts, cur, depth = [], "", 0
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth = max(0, depth - 1)
            if depth == 0 and text.startswith(", ", i):
                parts.append(cur)
                cur = ""
                i += 2
                continue
            cur += ch
            i += 1
        parts.append(cur)
        # greedy pack into lines of at most `width` characters
        lines, line = [], ""
        for w in parts:
            cand = w if not line else line + ", " + w
            if len(cand) <= width:
                line = cand
            else:
                if line:
                    lines.append(line)
                line = w
        if line:
            lines.append(line)
        return lines or [text]

    def _decode_matrix(self, arr: np.ndarray, vtype: str = None):
        """Detect tour/sub tour/route/assignment structure in a 0-1 matrix.

        Only ``bvar``/``btvar`` variables are decoded: a non-binary variable
        whose values happen to be 0/1 (e.g. a schedule of start times) is not
        an assignment or a tour.

        Returns a short summary printed below the matrix:
          - ``tour: 0->1->2->0`` when the matrix is a square successor matrix
            (at most one 1 per row and per column) containing a closed loop
            that visits every point and returns to its start,
          - ``sub tour: 0->1->3->0`` for a closed loop over only a subset
            of the points,
          - ``route: 0->1->2`` for an open chain shown alongside such a
            closed loop; segments are joined with ``, `` and labeled
            individually when they differ,
          - ``assignment: 0->1, 1->3`` when every row (else every column)
            carries at most one 1 — including successor matrices without
            any closed loop; rows holding several are grouped as
            ``0 -> [0, 2, 3]``,
          - ``covers: 0 -> [0, 1], 1->0`` for any other 0-1 matrix, listing
            per row which columns it covers,
          - ``None`` when the variable is not binary or the matrix is not 0-1.
        """
        if vtype not in ('bvar', 'btvar'):
            return None
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            return None
        if arr.size == 0:
            return None
        # solvers may return 0.9999999 for a binary 1 -> binarize on tolerance
        _tol = 1e-6
        near_zero = np.abs(arr) <= _tol
        near_one = np.abs(arr - 1.0) <= _tol
        if not np.all(near_zero | near_one):
            return None  # not a 0/1 matrix
        bin_arr = near_one.astype(np.int64)
        ones = np.argwhere(bin_arr)
        if len(ones) == 0:
            return None
        n_rows, n_cols = bin_arr.shape
        row_ones = np.count_nonzero(bin_arr, axis=1)
        col_ones = np.count_nonzero(bin_arr, axis=0)

        # --- tour / sub tour / route: square successor matrix ----------------
        if n_rows == n_cols and n_rows > 1 and np.all(row_ones <= 1) and np.all(col_ones <= 1):
            succ = {int(u): int(v) for u, v in ones}
            pred = {v: u for u, v in succ.items()}
            visited = set()
            segments = []  # list of (path, closed)
            # open chains start at nodes without a predecessor
            for start in sorted(succ):
                if start in pred or start in visited:
                    continue
                path = [start]
                visited.add(start)
                node = succ[start]
                while node is not None and node not in visited:
                    path.append(node)
                    visited.add(node)
                    node = succ.get(node)
                segments.append((path, False))
            # remaining nodes form closed loops
            for start in sorted(succ):
                if start in visited:
                    continue
                path = [start]
                visited.add(start)
                node = succ[start]
                while node != start and node not in visited:
                    path.append(node)
                    visited.add(node)
                    node = succ.get(node)
                segments.append((path, node == start))
            # only a matrix containing a closed loop reads as routing; open
            # chains alone are an assignment. a closed loop visiting every
            # point is a tour, one over only a subset a sub tour; open chains
            # in the same matrix are routes
            if any(closed and len(path) >= 2 for path, closed in segments):
                labeled = []
                for path, closed in segments:
                    if len(path) == 1:
                        continue  # self-loop is not a tour or a route
                    if closed:
                        chain = "->".join(str(n) for n in path) + f"->{path[0]}"
                        label = "tour" if len(path) == n_rows else "sub tour"
                    else:
                        chain = "->".join(str(n) for n in path)
                        label = "route"
                    labeled.append((label, chain))
                if all(label == labeled[0][0] for label, _ in labeled):
                    return labeled[0][0] + ": " + ", ".join(
                        chain for _, chain in labeled)
                return ", ".join(f"{label}: {chain}"
                                 for label, chain in labeled)

        # --- assignment --------------------------------------------------------
        if np.all(row_ones <= 1):
            return "assignment: " + ", ".join(f"{u}->{v}" for u, v in ones)
        if np.all(col_ones <= 1):
            # several columns may be assigned to one row -> group per row:
            #   assignment: 0 -> [0, 2, 3], 1 -> [4]
            return "assignment: " + self._format_row_groups(ones, n_rows)

        # --- plain membership: binary but neither route nor assignment --------
        # e.g. location x batch coverage: covers: 0 -> [0, 1], 1->0, ...
        return "covers: " + self._format_row_groups(ones, n_rows)

    @staticmethod
    def _format_row_groups(ones, n_rows):
        """Render 1-positions grouped by row: ``i->j`` for a single position
        and ``i -> [j, k, ...]`` when a row holds several."""
        parts = []
        for i in range(n_rows):
            cols = [int(v) for u, v in ones if int(u) == i]
            if not cols:
                continue
            if len(cols) == 1:
                parts.append(f"{i}->{cols[0]}")
            else:
                parts.append(f"{i} -> [{', '.join(str(c) for c in cols)}]")
        return ", ".join(parts)

    def render_tensor_blocks(
        self,
        label: str,
        tensor: Any,
        additional_text: str = "",
        decimal_places: int = 4
    ) -> list:
        if isinstance(tensor, dict):
            # render per key so dictionaries over partial dimensions show as
            # their pieces (``label[0] = [1, 2]``, ``label[3] = 4``)
            if not tensor:
                return [f"{label} = {{}}{additional_text}"]
            blocks = []
            for k, v in tensor.items():
                ks = str(k).replace("(", "[").replace(")", "]")
                if "[" not in ks:
                    ks = f"[{ks}]"
                # scalar zero leaves are suppressed exactly like
                # _print_dict_element does in element mode; non-numeric
                # values and containers are always rendered
                is_scalar_leaf = ((isinstance(v, np.ndarray) and v.ndim == 0)
                                  or not isinstance(v, (dict, list, tuple, np.ndarray)))
                if is_scalar_leaf:
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        pass  # non-numeric leaf: shown as-is
                    else:
                        if abs(fv) < 10 ** (-decimal_places):
                            continue
                blocks.extend(self.render_tensor_blocks(f"{label}{ks}", v, "", decimal_places))
            if not blocks:
                # every leaf was suppressed: say so instead of dropping the
                # entry (print_packed_tensors skips empty groups)
                return [f"{label} = all zeros{additional_text}"]
            blocks[0] = blocks[0] + additional_text
            return blocks
        arr = np.array(tensor)
        inner = self.width - 4

        if arr.ndim == 0:
            val_str = self._format_value(float(arr), decimal_places)
            return [f"{label} = {val_str}{additional_text}"]

        nnz = int(np.count_nonzero(arr))
        total = arr.size
        if nnz == 0:
            return [f"{label} = all zeros{additional_text}"]

        if arr.ndim >= 2:
            if arr.ndim == 2:
                slices = [((), arr)]
            else:
                trail = arr.shape[2:]
                slices = []
                for idx in np.ndindex(*trail):
                    slicer = (slice(None), slice(None)) + idx
                    slices.append((idx, arr[slicer]))

            slice_blocks = []
            for slice_idx, (idx, slice_2d) in enumerate(slices):
                if not np.any(np.abs(slice_2d) >= 10 ** (-decimal_places)):
                    continue  # hide slices where every value shows as zero
                if len(slices) == 1:
                    var_name = label
                else:
                    trail_label = ",".join(str(i) for i in idx)
                    var_name = f"{label}[...,{trail_label}]"

                col_widths = [0] * slice_2d.shape[1]
                for row_idx in range(slice_2d.shape[0]):
                    for ci, val in enumerate(slice_2d[row_idx]):
                        fv = float(val.item()) if hasattr(val, 'item') else float(val)
                        fs = self._format_value(fv, decimal_places)
                        col_widths[ci] = max(col_widths[ci], len(fs))

                row_is_zero = [np.all(np.abs(slice_2d[r]) < 1e-10) for r in range(slice_2d.shape[0])]

                row_strs = []
                r = 0
                while r < slice_2d.shape[0]:
                    if row_is_zero[r]:
                        count = 0
                        while r < slice_2d.shape[0] and row_is_zero[r]:
                            count += 1
                            r += 1
                        if count >= 3:
                            row_strs.append(f"[zeros\u00d7{count}]")
                        else:
                            for _ in range(count):
                                cells = [self._format_value(0.0, decimal_places).rjust(col_widths[ci]) for ci in range(slice_2d.shape[1])]
                                row_strs.append("[" + ", ".join(cells) + "]")
                    else:
                        formatted = []
                        for ci, val in enumerate(slice_2d[r]):
                            fv = float(val.item()) if hasattr(val, 'item') else float(val)
                            formatted.append(self._format_value(fv, decimal_places))
                        cells = [fs.rjust(col_widths[ci]) for ci, fs in enumerate(formatted)]
                        row_strs.append("[" + ", ".join(cells) + "]")
                        r += 1

                prefix = f"{var_name} = "
                indent = " " * len(prefix)
                cont_indent = " " * (len(prefix) + 1)
                parts = []
                for rs in row_strs:
                    # One matrix row per line, stacked like a table.
                    full_line = (prefix if not parts else indent) + rs
                    if len(full_line) <= inner:
                        parts.append(full_line)
                        continue
                    cells = rs[1:-1].split(", ")
                    line = (prefix if not parts else indent) + "["
                    for ci, cell in enumerate(cells):
                        is_last_cell = (ci == len(cells) - 1)
                        sep = ", " if not line.endswith("[") else ""
                        test = line + sep + cell
                        if is_last_cell:
                            max_len = inner
                        else:
                            max_len = inner - 2
                        if len(test) <= max_len:
                            line = test
                        else:
                            if not line.endswith("["):
                                parts.append(line + ",")
                                line = cont_indent + cell
                            else:
                                line = test
                    if not line.endswith("]"):
                        line += "]"
                    parts.append(line)
                # route/assignment summary below the matrix (tensor mode only:
                # element mode shows raw indexed entries without annotations)
                annotation = self._decode_matrix(
                    slice_2d, getattr(self, 'var_types', {}).get(label)
                )
                if annotation:
                    for chunk in self._chunk_annotation(annotation, inner - len(prefix)):
                        parts.append((indent if parts else prefix) + chunk)
                for pi in range(len(parts)):
                    if pi == len(parts) - 1 and additional_text:
                        parts[pi] = parts[pi] + additional_text
                    slice_blocks.append(parts[pi])

            if not slice_blocks:
                return [f"{label} = all zeros{additional_text}"]

            return slice_blocks

        def custom_formatter(x):
            fv = float(x)
            if fv == 0 or (abs(fv) < 0.5 * 10 ** (-decimal_places)):
                return "0"
            if fv.is_integer():
                return str(int(fv))
            s = f"{fv:.{decimal_places}f}"
            s = s.rstrip("0").rstrip(".")
            if len(s.replace("-", "").replace(".", "")) > self.sci_thresh:
                sci = f"{fv:.{decimal_places}e}"
                m, e = sci.split("e")
                m = m.rstrip("0").rstrip(".")
                return f"{m}e{e}"
            return s
        formatter = {"float_kind": custom_formatter}
        txt = np.array2string(arr, separator=", ", prefix="", formatter=formatter, max_line_width=100000, threshold=10000, edgeitems=10000)
        full = f"{label} = {txt}{additional_text}"
        indent = " " * (len(label) + 4)
        lines = []
        rest = full
        while rest:
            if len(rest) <= inner:
                lines.append(rest)
                break
            best = rest[:inner].rfind(", ")
            bracket = rest[:inner].rfind("]")
            cut = max(best, bracket)
            if cut < inner // 4:
                cut = inner
            else:
                cut = cut + 2
            lines.append(rest[:cut])
            rest = indent + rest[cut:]
        return lines

    def _emit_block(self, line, indent):
        inner = self.width - 4
        if len(line) <= inner:
            self._emit(f"{self._border} {line.ljust(inner)} {self._border}", "row")
        else:
            stripped = line.lstrip()
            content_indent = len(line) - len(stripped)
            if content_indent == 0 and indent:
                # hang continuation lines under ``label = value`` instead of
                # restarting at the left border
                content_indent = len(indent)
            if stripped.startswith("["):
                content_indent += 1
            # keep room for continuation content on every wrapped line
            content_indent = min(content_indent, max(0, inner - 20))
            chunks = []
            rest = line
            first = True
            while rest:
                cap = inner if first else inner - content_indent
                first = False
                if len(rest) <= cap:
                    chunks.append(rest)
                    rest = ""
                else:
                    split_at = cap
                    last_comma = rest[:cap].rfind(", ")
                    last_bracket = rest[:cap].rfind("]")
                    best = max(last_comma, last_bracket)
                    if best > cap // 3:
                        split_at = min(best + 2, cap)
                    # don't start the next line with a dangling ", "
                    if rest[split_at:split_at + 2] == ", ":
                        alt = rest[:split_at].rfind(", ")
                        if alt > cap // 3:
                            split_at = alt + 2
                    chunks.append(rest[:split_at])
                    rest = rest[split_at:]
            for i, chunk in enumerate(chunks):
                if i == 0:
                    self._emit(f"{self._border} {chunk.ljust(inner)} {self._border}", "row")
                else:
                    # cap keeps this within inner — never slice content away
                    padded = (" " * content_indent + chunk).ljust(inner)
                    self._emit(f"{self._border} {padded} {self._border}", "row")

    def print_packed_tensors(
        self,
        variables: list,
        additional_text: str = "",
        decimal_places: int = 4
    ) -> None:
        inner = self.width - 4
        gap = 3

        var_groups = []
        for label, tensor in variables:
            try:
                blocks = self.render_tensor_blocks(label, tensor, "", decimal_places)
            except Exception:
                # never drop an entry — one failure would otherwise abort the
                # rest of the section: fall back to its string form
                blocks = [f"{label} = {tensor}"]
            if blocks:
                var_groups.append(blocks)

        if not var_groups:
            return

        if len(var_groups) == 1 and len(var_groups[0]) == 1:
            indent = " " * len(var_groups[0][0].split(" = ", 1)[0] + " = ")
            self._emit_block(var_groups[0][0], indent)
            return

        packed_rows = []
        current_row = []
        current_width = 0

        for group in var_groups:
            group_width = max(len(b) for b in group)
            if current_row and current_width + gap + group_width <= inner:
                current_row.append(group)
                current_width += gap + group_width
            elif not current_row:
                current_row.append(group)
                current_width = group_width
            else:
                packed_rows.append(current_row)
                current_row = [group]
                current_width = group_width
        if current_row:
            packed_rows.append(current_row)

        for row_groups in packed_rows:
            max_lines = max(len(g) for g in row_groups)
            for line_idx in range(max_lines):
                # Normalize every column to its group's widest block so side-by-side
                # groups stay aligned even when a group has fewer lines (filler rows)
                # or blocks of differing lengths.
                widths = [max(len(b) for b in g) for g in row_groups]
                line_parts = []
                for gi, g in enumerate(row_groups):
                    part = g[line_idx] if line_idx < len(g) else ""
                    if gi < len(row_groups) - 1:
                        part = part.ljust(widths[gi])
                    line_parts.append(part)
                line = (" " * gap).join(line_parts)
                indent = " " * len(line_parts[0].split(" = ", 1)[0] + " = ") if " = " in line_parts[0] else " " * 4
                self._emit_block(line, indent)

    def print_tensor(
        self,
        label: str,
        tensor: Any,
        additional_text: str = "",
        decimal_places: int = 4
    ) -> None:
        self.print_packed_tensors([(label, tensor)], additional_text, decimal_places)

    def print_pandas_df(
        self,
        label: str,
        df: pl.DataFrame,
        columns: Sequence[str] = None,
        additional_text: str = "",
        decimal_places: int = 4,
        max_space_between_elements: int = 4,
        extra_spaces: int = 0
    ) -> None:
        cols = columns if columns is not None else df.columns
        for idx, row in enumerate(df[cols].iter_rows(named=True)):
            strs = [self._format_value(row[c], decimal_places) for c in df[cols].columns]
            self.clear_columns(
                strs,
                f"{label}[{idx}]",
                max_space_between_elements=max_space_between_elements,
                extra_spaces=extra_spaces
            )
        if additional_text:
            inner = self.width - 4
            self._emit(
                f"{self._border} "
                f"{additional_text[:inner].ljust(inner)} "
                f"{self._border}", "row"
            )


Box = report


def left_align(input, box_width=78, rt=False):
    max_text = box_width - 4
    if max_text <= 0:
        text = ""
    else:
        text = str(input)[:max_text] if len(str(input)) > max_text else str(input)

    if rt:
        return "\u2502" + " " + text.ljust(max_text) + " " + "\u2502"
    else:
        print("\u2502" + " " + text.ljust(max_text) + " " + "\u2502")


def format_string(input, length=8, ensure_length=False):

    if isinstance(input, str):
        return input

    if input is None:
        formatted_str = "None"

    elif isinstance(input, int):
        if abs(input) <= 10000:
            formatted_str = f"{input:.0f}"
        else:
            formatted_str = f"{input:.2e}"

    elif isinstance(input, float):
        if abs(input) <= 10000:
            formatted_str = f"{input:.2f}"
        else:
            formatted_str = f"{input:.2e}"

    elif isinstance(input, dict):
        formatted_str = ", ".join(
            f"{k}: {format_string(v, length, ensure_length)}" for k, v in input.items()
        )

    else:
        formatted_str = str(input)

    if ensure_length and len(formatted_str) < length:
        formatted_str += " " * (length - len(formatted_str))

    return formatted_str


def _colorize_obj_values(values, directions=None, all_alternatives=None, alt_index=None):
    """Return a list of colorama-colored format strings for objective values.

    For each objective column, best gets GREEN, worst gets RED,
    middle (if any) gets YELLOW. Direction determines which end is best:
    'min' → lowest is best; 'max' → highest is best.

    Parameters
    ----------
    values : list
        Values for ONE alternative (one row).
    directions : list or None
        Direction per objective column ('min'/'max').
    all_alternatives : numpy array or None
        Full (n_alternatives x n_objectives) matrix, used to compare across alternatives.
    alt_index : int or None
        Row index of this alternative in all_alternatives.
    """
    from colorama import Fore
    n_obj = len(values)
    colored = []
    if n_obj == 0:
        return colored

    vals = [float(v) for v in values]
    if directions is None:
        directions = [None] * n_obj

    for j in range(n_obj):
        raw = format_string(vals[j])

        col_vals = None
        if all_alternatives is not None and alt_index is not None:
            try:
                col_vals = [float(all_alternatives[i][j]) for i in range(all_alternatives.shape[0])]
            except Exception:
                pass

        if col_vals is None or len(col_vals) <= 1:
            colored.append(raw)
            continue

        same = all(abs(col_vals[k] - col_vals[0]) < 1e-10 for k in range(1, len(col_vals)))
        if same:
            colored.append(raw)
            continue

        best_val = min(col_vals) if directions[j] == 'min' else max(col_vals)
        worst_val = max(col_vals) if directions[j] == 'min' else min(col_vals)

        if abs(vals[j] - best_val) < 1e-10:
            colored.append(f"{Fore.GREEN}{raw}{Fore.RESET}")
        elif abs(vals[j] - worst_val) < 1e-10:
            colored.append(f"{Fore.RED}{raw}{Fore.RESET}")
        else:
            colored.append(f"{Fore.YELLOW}{raw}{Fore.RESET}")

    return colored


def format_text(input_str, length=8, ensure_length=False):
    if input_str is None:
        formatted_str = "None"
    else:
        formatted_str = input_str

    if ensure_length:
        if len(formatted_str) < length:
            formatted_str += " " * (length - len(formatted_str))
        elif len(formatted_str) > length:
            formatted_str = formatted_str[: length - 3] + "..."

    return formatted_str


def center(input, box_width=78):
    max_text = box_width - 4
    if max_text <= 0:
        text = ""
    else:
        text = str(input)[:max_text] if len(str(input)) > max_text else str(input)
    print("\u2502" + " " + text.center(max_text) + " " + "\u2502")


def tline(box_width=78):
    print("\u250c" + "\u2500" * box_width + "\u2510")


def tline_text(input, box_width=78):
    max_text = box_width - 3
    if max_text <= 0:
        text = ""
    else:
        text = str(input)[:max_text] if len(str(input)) > max_text else str(input)
    print("\u250c\u2500 " + text + " " + "\u2500" * (box_width - len(text) - 3) + "\u2510")


def bline_text(input, box_width=78):
    max_text = box_width - 3
    if max_text <= 0:
        text = ""
    else:
        text = str(input)[:max_text] if len(str(input)) > max_text else str(input)
    print("\u2514" + text + " " + "\u2500" * (box_width - len(text) - 3) + "\u2518")


def empty_line(box_width=78):
    print("\u2502" + " " * box_width + "\u2502")


def bline(box_width=78):
    print("\u2514" + "\u2500" * box_width + "\u2518")


def hline(box_width=78):
    print("\u251c" + "\u2500" * box_width + "\u2524")


def whline(box_width=78):
    print("+ " + "." * box_width + "\u2524")


def hrule(box_width=78):
    print("\u251c" + "=" * box_width + "\u2524")


def vspace():
    print()


def two_column(input1, input2, box_width=78):
    max_each = (box_width - 4) // 2
    s1 = str(input1)[:max_each] if len(str(input1)) > max_each else str(input1)
    s2 = str(input2)[:max_each] if len(str(input2)) > max_each else str(input2)
    padding = box_width - len(s1) - len(s2) - 2
    if padding < 0:
        padding = 0
    print("\u2502 " + s1 + " " * padding + s2 + " \u2502")


def three_column(input1, input2, input3, box_width=78, underline=False):
    max_each = (box_width - 8) // 3
    s1 = str(input1)[:max_each] if len(str(input1)) > max_each else str(input1)
    s2 = str(input2)[:max_each] if len(str(input2)) > max_each else str(input2)
    s3 = str(input3)[:max_each] if len(str(input3)) > max_each else str(input3)
    total_padding = box_width - len(s1) - len(s2) - len(s3) - 6
    if total_padding < 0:
        total_padding = 0
    padding1 = total_padding // 2
    padding2 = total_padding - padding1

    if underline:
        s1 = f"\033[4m{s1}\033[0m"
        s2 = f"\033[4m{s2}\033[0m"
        s3 = f"\033[4m{s3}\033[0m"

    print(
        "\u2502 "
        + s1
        + " " * padding1
        + "  "
        + s2
        + " " * padding2
        + "  "
        + s3
        + " \u2502"
    )


def list_three_column(input_list, box_width=78, underline=False):
    max_each = (box_width - 6) // 3
    column1_width = min(max(len(str(x[0])) for x in input_list) + 2, max_each)
    column2_width = min(max(len(str(x[1])) for x in input_list) + 2, max_each)
    column3_width = min(max(len(str(x[2])) for x in input_list) + 2, max_each)

    space_left = box_width - column1_width - column2_width - column3_width - 2
    if space_left < 0:
        space_left = 0
    padding1 = space_left // 2
    padding2 = space_left - padding1

    for row in input_list:
        if row[2] != 0:
            input1, input2, input3 = row
            if underline:
                input1 = f"\033[4m{input1}\033[0m"
                input2 = f"\033[4m{input2}\033[0m"
                input3 = f"\033[4m{input3}\033[0m"
            print(
                f"\u2502 {input1:<{column1_width}}{padding1 * ' '}{input2:<{column2_width}}{padding2 * ' '}{input3:<{column3_width}} \u2502"
            )


def boxed(text, box_width=78):
    import textwrap
    lines = textwrap.wrap(text, width=box_width - 4)
    for line in lines:
        left_align(line, box_width)

def right_align(input, box_width=78):
    max_text = box_width - 4
    if max_text <= 0:
        text = ""
    else:
        text = str(input)[-max_text:] if len(str(input)) > max_text else str(input)
    print("\u2502" + " " + text.rjust(max_text) + " " + "\u2502")

def feature_print(type, input):
    if input[0] > 0:
        three_column(type, f"{input[0]}", f"{input[1]}")


def objective_print(type, input):
    if input[0] > 0:
        three_column(type, "-", f"{input[1]}")


def constraint_print(type, input):
    if input[0] > 0:
        three_column(type, f"{input[0]}", f"{input[1]}")


def status_row_print(ObjectivesDirections, status, box_width=78):
    if len(ObjectivesDirections) != 1 and ObjectivesDirections[0] != "nan":
        row = (
            "\u2502 "
            + "Status: "
            + " " * (len(status[0]) - len("Status: "))
            + " "
            * (box_width - 10 * len(ObjectivesDirections) + 1 - len(str(status[0])) - 3)
        )
        for j in range(len(ObjectivesDirections)):
            obj_row = ObjectivesDirections[j]
            row += " " * (10 - len(obj_row)) + obj_row
        print(row + " \u2502")

    else:
        row = (
            "\u2502 "
            + "Status: "
            + " " * (len(str(status)) - len("Status: "))
            + " "
            * (box_width - 10 * len(ObjectivesDirections) + 1 - len(str(status)) - 3)
        )
        for j in range(len(ObjectivesDirections)):
            obj_row = ObjectivesDirections[j]
            row += " " * (10 - len(obj_row)) + obj_row

        if len(row + " \u2502") == box_width + 2:
            print(row + " \u2502")

        elif len(row + " \u2502") < box_width + 2:
            row = (
                "\u2502 "
                + "Status: "
                + " " * (len(str(status)) - len("Status: "))
                + " "
                * (
                    box_width
                    - 10 * len(ObjectivesDirections)
                    + 1
                    - len(str(status))
                    - 3
                )
            )
            for j in range(len(ObjectivesDirections)):
                obj_row = ObjectivesDirections[j]
                row += " " * (10 - len(obj_row)) + obj_row
            print(row + " \u2502")

        else:
            row = (
                "\u2502 "
                + "Status: "
                + " " * (len(str(status)) - len("Status: "))
                + " "
                * (box_width - 10 * len(ObjectivesDirections) - len(str(status)) - 3)
            )
            for j in range(len(ObjectivesDirections)):
                obj_row = ObjectivesDirections[j]
                row += " " * (10 - len(obj_row)) + obj_row
            print(row + " \u2502")


def solution_print(
    ObjectivesDirections, status, get_obj, get_payoff=None, box_width=78
):

    if len(ObjectivesDirections) != 1:
        if status[0] != "infeasible (constrained)":
            for i in range(len(status)):
                row = (
                    "\u2502 "
                    + str(status[i])
                    + " "
                    * (
                        box_width
                        - 10 * len(ObjectivesDirections)
                        + 1
                        - len(str(status[i]))
                        - 3
                    )
                )
                obj_row = get_obj[i]
                for j in range(len(obj_row)):
                    num_str = format_string(obj_row[j])
                    row += " " * (10 - len(num_str)) + num_str
                print(row + " \u2502")

            for j in range(len(ObjectivesDirections)):
                row = (
                    "\u2502 "
                    + str(f"payoff {j}")
                    + " "
                    * (
                        box_width
                        - 10 * len(ObjectivesDirections)
                        + 1
                        - len(str(f"payoff {j}"))
                        - 3
                    )
                )
                for k in range(len(ObjectivesDirections)):
                    num_str = format_string(get_payoff[j, k])
                    row += " " * (10 - len(num_str)) + num_str
                print(row + " \u2502")

            row = (
                "\u2502 "
                + str("max")
                + " "
                * (box_width - 10 * len(ObjectivesDirections) + 1 - len(str("max")) - 3)
            )
            for j in range(len(ObjectivesDirections)):
                num_str = format_string(np.max(get_obj[:, j]))
                row += " " * (10 - len(num_str)) + num_str
            print(row + " \u2502")

            row = (
                "\u2502 "
                + str("ave")
                + " "
                * (box_width - 10 * len(ObjectivesDirections) + 1 - len(str("ave")) - 3)
            )
            for j in range(len(ObjectivesDirections)):
                num_str = format_string(np.average(get_obj[:, j]))
                row += " " * (10 - len(num_str)) + num_str
            print(row + " \u2502")

            row = (
                "\u2502 "
                + str("std")
                + " "
                * (box_width - 10 * len(ObjectivesDirections) + 1 - len(str("std")) - 3)
            )
            for j in range(len(ObjectivesDirections)):
                num_str = format_string(np.std(get_obj[:, j]))
                row += " " * (10 - len(num_str)) + num_str
            print(row + " \u2502")

            row = (
                "\u2502 "
                + str("min")
                + " "
                * (box_width - 10 * len(ObjectivesDirections) + 1 - len(str("min")) - 3)
            )
            for j in range(len(ObjectivesDirections)):
                num_str = format_string(np.min(get_obj[:, j]))
                row += " " * (10 - len(num_str)) + num_str
            print(row + " \u2502")

    else:
        row = (
            "\u2502 "
            + str(status)
            + " "
            * (box_width - 9 * len(ObjectivesDirections) + 1 - len(str(status)) - 3)
        )
        obj_row = get_obj
        num_str = format_string(obj_row)
        row += " " * (9 - len(num_str)) + num_str
        print(row + " \u2502")


def metrics_print(
    ObjectivesDirections,
    show_all_metrics,
    get_obj,
    calculated_indicators,
    start=0,
    end=0,
    length=None,
    box_width=78,
):

    hour, min, sec = calculate_time_difference(start, end, length)

    try:
        if len(ObjectivesDirections) != 1:
            if show_all_metrics and len(get_obj) != 0:
                for key, label in [
                    ("gd", "GD (min)"),
                    ("gdp", "GDP (min)"),
                    ("igd", "IGD (min)"),
                    ("igdp", "IGDP (min)"),
                    ("ms", "MS (max)"),
                    ("sp", "SP (min)"),
                    ("hv", "HV (max)"),
                ]:
                    value = calculated_indicators.get(key)
                    if value is not None:
                        two_column(label, format_string(value))
    except Exception as e:
        center(f"No special metric is calculatable.")

    if length == None:
        two_column("CPT (microseconds)", format_string((end - start) * 10**6))
    else:
        two_column("CPT (microseconds)", format_string((length) * 10**6))

    two_column("CPT (hour:min:sec)", "%02d:%02d:%02d" % (hour, min, sec))


def calculate_time_difference(start=0, end=0, length=None):
    if length is None:
        delta = round((end - start), 3)
    else:
        delta = round(length, 3)

    hour = delta % (24 * 3600) // 3600
    minute = delta % (24 * 3600) % 3600 // 60
    second = delta % (24 * 3600) % 3600 % 60

    return hour, minute, second


def _metric_label(metric, value, problem_size=0):
    if value == "-":
        return "-"
    try:
        v = float(value)
    except (ValueError, TypeError):
        return "-"
    if metric == "STG":
        if v == 0: return "Fresh"
        if v < 0.3: return "Moving"
        if v < 0.7: return "Stalling"
        return "Stuck"
    elif metric == "CVR":
        if v == 0: return "Aligned"
        if v < 0.3: return "Weak"
        if v < 0.7: return "Moderate"
        return "Strong"
    elif metric == "IMP":
        if v < 0: return "Declining"
        if v < 0.01: return "Stagnant"
        if v < 0.1: return "Slow"
        if v < 0.5: return "Moderate"
        return "Fast"
    elif metric == "DIV":
        if v == 0: return "Uniform"
        if v < 0.1: return "Narrow"
        if v < 0.5: return "Balanced"
        return "Diverse"
    elif metric == "OGR":
        if v == 0: return "Optimal"
        if v < 0.01: return "Proven"
        if v < 0.1: return "Tight"
        if v < 0.5: return "Loose"
        return "Wide"
    elif metric == "PVR":
        if v == 0: return "None"
        if v < 0.5: return "Lean"
        if v <= 1.5: return "Balanced"
        if v < 3: return "Heavy"
        return "Overloaded"
    elif metric == "NZR":
        if v == 0: return "Zero"
        if v < 0.1: return "Sparse"
        if v < 0.5: return "Balanced"
        return "Dense"
    elif metric == "MGT":
        if v < 1: return "Rapid"
        if v < 60: return "Smooth"
        if problem_size > 100: return "Smooth"
        if problem_size > 20: return "Slow"
        return "Complex"
    elif metric == "CPT":
        if v < 1: return "Rapid"
        if v < 60: return "Smooth"
        if problem_size > 100: return "Smooth"
        if problem_size > 20: return "Slow"
        return "Complex"
    return "-"


def format_time_duration(seconds_value, name=""):
    if seconds_value < 0.001:
        return name + f"{int(seconds_value * 1e6)} \u03bcs"
    elif seconds_value < 1:
        return name + f"{int(seconds_value * 1000)} ms"
    elif seconds_value < 60:
        return name + f"{seconds_value:.2f} s"
    elif seconds_value < 3600:
        m = int(seconds_value // 60)
        s = int(seconds_value % 60)
        return name + f"{m}m {s:02d}s"
    else:
        h = int(seconds_value // 3600)
        m = int((seconds_value % 3600) // 60)
        s = int(seconds_value % 60)
        return name + f"{h}h {m:02d}m {s:02d}s"


def _compute_pareto_metrics(objective_values):
    import numpy as np
    if objective_values is None or not isinstance(objective_values, np.ndarray):
        return {"hypervolume": None, "spacing": None, "spread": None, "gd": None, "igd": None, "n_solutions": 0}
    if objective_values.ndim == 1:
        objective_values = objective_values.reshape(1, -1)
    n_sols, n_obj = objective_values.shape
    if n_sols == 0:
        return {"hypervolume": None, "spacing": None, "spread": None, "gd": None, "igd": None, "n_solutions": 0}
    result = {"n_solutions": n_sols}
    nadir = np.max(objective_values, axis=0)
    ref = nadir * 1.1 + 1e-6
    non_dominated_mask = np.ones(n_sols, dtype=bool)
    for i in range(n_sols):
        for j in range(n_sols):
            if i != j and np.all(objective_values[j] <= objective_values[i]) and np.any(objective_values[j] < objective_values[i]):
                non_dominated_mask[i] = False
                break
    nd_FRONT = objective_values[non_dominated_mask]
    if len(nd_FRONT) == 0:
        result["hypervolume"] = 0.0
    else:
        try:
            hv = 0.0
            sorted_idx = np.argsort(nd_FRONT[:, 0])
            sorted_front = nd_FRONT[sorted_idx]
            for k in range(len(sorted_front)):
                width_k = (ref[0] - sorted_front[k, 0]) if k == 0 else (sorted_front[k - 1, 0] - sorted_front[k, 0])
                height_k = ref[1] - sorted_front[k, 1] if n_obj > 1 else 1.0
                hv += max(0, width_k) * max(0, height_k)
            result["hypervolume"] = float(hv)
        except Exception:
            result["hypervolume"] = None
    if n_sols <= 1:
        result["spacing"] = 0.0
    else:
        try:
            dists = []
            for i in range(n_sols):
                min_d = float('inf')
                for j in range(n_sols):
                    if i != j:
                        d = np.sum(np.abs(objective_values[i] - objective_values[j]))
                        min_d = min(min_d, d)
                dists.append(min_d)
            mean_d = np.mean(dists)
            result["spacing"] = float(np.sqrt(np.mean((dists - mean_d) ** 2))) if mean_d > 0 else 0.0
        except Exception:
            result["spacing"] = None
    try:
        ideal = np.min(objective_values, axis=0)
        nadir_arr = np.max(objective_values, axis=0)
        range_arr = nadir_arr - ideal
        range_arr[range_arr == 0] = 1.0
        normalized = (objective_values - ideal) / range_arr
        if n_sols > 1:
            sorted_norm = normalized[np.argsort(normalized[:, 0])]
            gaps = np.linalg.norm(sorted_norm[1:] - sorted_norm[:-1], axis=1)
            sum_gaps = np.max(gaps) + np.min(gaps)
            result["spread"] = float(1.0 - (np.max(gaps) - np.min(gaps)) / sum_gaps) if sum_gaps > 0 else 0.0
        else:
            result["spread"] = 0.0
    except Exception:
        result["spread"] = None
    try:
        if len(nd_FRONT) > 0 and n_sols > 1:
            gd_vals = []
            for i in range(n_sols):
                diffs = nd_FRONT - objective_values[i]
                dists_to_nd = np.min(np.sqrt(np.sum(diffs ** 2, axis=1)))
                gd_vals.append(dists_to_nd ** 2)
            result["gd"] = float(np.sqrt(np.mean(gd_vals)))
        else:
            result["gd"] = 0.0
    except Exception:
        result["gd"] = None
    try:
        if len(nd_FRONT) > 0 and n_sols > 1:
            igd_vals = []
            for i in range(len(nd_FRONT)):
                diffs = objective_values - nd_FRONT[i]
                dists_to_all = np.min(np.sqrt(np.sum(diffs ** 2, axis=1)))
                igd_vals.append(dists_to_all ** 2)
            result["igd"] = float(np.sqrt(np.mean(igd_vals)))
        else:
            result["igd"] = 0.0
    except Exception:
        result["igd"] = None
    try:
        if n_sols > 1 and n_obj >= 2:
            from itertools import combinations
            radeski_vals = []
            for dim_pair in combinations(range(n_obj), 2):
                d1, d2 = dim_pair
                proj = objective_values[:, [d1, d2]]
                proj_sorted = proj[np.argsort(proj[:, 0])]
                edge_len = np.linalg.norm(proj_sorted[-1] - proj_sorted[0])
                if edge_len > 0:
                    interior_lengths = np.linalg.norm(np.diff(proj_sorted, axis=0), axis=1)
                    n_interior = len(interior_lengths)
                    if n_interior > 0:
                        radeski_vals.append(float(np.sum(interior_lengths) / (edge_len * n_interior)))
            result["cluster_compactness"] = float(np.mean(radeski_vals)) if radeski_vals else None
        else:
            result["cluster_compactness"] = None
    except Exception:
        result["cluster_compactness"] = None
    return result


def detect_system_info(width=80):
    import platform
    import subprocess
    os_name = platform.system()
    if os_name == "Windows":
        release, version, _, _ = platform.win32_ver()
        build = int(version.split(".")[2]) if version and len(version.split(".")) >= 3 else 0
        os_info = "Windows 11" if release == "10" and build >= 22000 else f"Windows {release}"
    elif os_name == "Linux":
        os_info = f"Linux {platform.release()}"
    elif os_name == "Darwin":
        os_info = f"macOS {platform.mac_ver()[0]}"
    else:
        os_info = f"{os_name} {platform.release()}"

    try:
        import cpuinfo
        ci = cpuinfo.get_cpu_info()
        arch_raw = ci.get('arch_string_raw', '') or ci.get('arch', '')
        arch_l = arch_raw.lower()
        if 'arm' in arch_l:
            arch = 'ARM'
        elif 'x86_64' in arch_l or 'amd64' in arch_l:
            arch = 'x64'
        elif '386' in arch_l or 'i386' in arch_l:
            arch = 'x86'
        else:
            m = platform.machine().lower()
            if m in ('amd64', 'x86_64', 'x64'):
                arch = 'x64'
            elif m in ('i386', 'i686', 'x86'):
                arch = 'x86'
            elif 'arm' in m or 'aarch64' in m:
                arch = 'ARM'
            else:
                arch = m
        raw = ci.get('brand_raw', '').strip()
        low = raw.lower()
        if 'intel' in low:
            cpu_brand = 'Intel'
        elif 'amd' in low:
            cpu_brand = 'AMD'
        elif 'qualcomm' in low or 'snapdragon' in low:
            cpu_brand = 'Qualcomm'
        elif 'apple' in low:
            cpu_brand = 'Apple'
        else:
            cpu_brand = raw.split()[0] if raw else 'CPU'
        cpu_spec = raw.replace(cpu_brand, '').split('@')[0].strip()
    except Exception:
        arch = platform.machine()
        cpu_brand = 'CPU'
        cpu_spec = ''

    try:
        import psutil
        ram_gb = int(np.round(psutil.virtual_memory().total / (1024.0 ** 3)))
    except Exception:
        ram_gb = 0

    gpu_entries = []
    if os_name == "Windows":
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name, AdapterRAM | "
                "ForEach-Object { \"$($_.Name)|$($_.AdapterRAM)\" }"],
                capture_output=True, text=True, timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if not line or "|" not in line:
                    continue
                name, vram_bytes_str = line.split("|", 1)
                name = name.strip()
                try:
                    vram_bytes = int(vram_bytes_str.strip())
                except (ValueError, TypeError):
                    vram_bytes = 0
                if not name:
                    continue
                if vram_bytes > 0:
                    vram_gb = int(round(vram_bytes / (1024 ** 3)))
                    gpu_entries.append(f"{name} ({vram_gb} GB)")
                else:
                    gpu_entries.append(name)
        except Exception:
            pass
    elif os_name == "Linux":
        try:
            result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=10)
            for line in result.stdout.splitlines():
                lower = line.lower()
                if "vga" in lower or "3d" in lower or "display" in lower:
                    parts = line.split(": ", 1)
                    name = parts[1].strip() if len(parts) > 1 else line.strip()
                    gpu_entries.append(name)
        except Exception:
            pass
        import glob as globmod
        for i, entry in enumerate(gpu_entries):
            vram_paths = globmod.glob(f"/sys/class/drm/card{i}-device/mem_info_vram_total")
            if not vram_paths:
                vram_paths = globmod.glob(f"/sys/class/drm/card{i}/device/mem_info_vram_total")
            for vp in vram_paths:
                try:
                    with open(vp) as f:
                        vram_bytes = int(f.read().strip())
                    vram_gb = int(round(vram_bytes / (1024 ** 3)))
                    if vram_gb > 0:
                        gpu_entries[i] = f"{entry} ({vram_gb} GB)"
                except (ValueError, OSError):
                    pass
    elif os_name == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=10
            )
            current_gpu_name = None
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("Chipset Model:") or stripped.startswith("Chip Model:"):
                    current_gpu_name = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("VRAM (Total):") and current_gpu_name:
                    vram_str = stripped.split(":", 1)[1].strip()
                    gpu_entries.append(f"{current_gpu_name} ({vram_str})")
                    current_gpu_name = None
            if current_gpu_name and current_gpu_name not in gpu_entries:
                gpu_entries.append(current_gpu_name)
        except Exception:
            pass
    if not gpu_entries:
        if cpu_brand == 'Intel':
            gpu_entries.append('Intel Integrated Graphics (shared)')
        elif cpu_brand == 'AMD':
            gpu_entries.append('AMD Integrated Graphics (shared)')
        elif cpu_brand == 'Qualcomm':
            gpu_entries.append('Qualcomm Adreno Integrated Graphics (shared)')
        elif cpu_brand == 'Apple':
            gpu_entries.append('Apple Integrated GPU')
    report = f"OS: {os_info} | Arch: {arch} | CPU: {cpu_brand} {cpu_spec}, {ram_gb} GB RAM | GPU: {', '.join(gpu_entries)}"
    if len(report) > width - 5:
        short = f"OS: {os_info} | Arch: {arch} | CPU: {cpu_brand} {cpu_spec}, {ram_gb}GB"
        if len(short) > width - 5:
            short = f"OS: {os_info} | Arch: {arch} | CPU: {cpu_brand} | RAM: {ram_gb}GB"
            return short
        return short
    return report


def detect_system_info_verbose(width=78):
    import platform
    import subprocess
    try:
        import psutil
        import cpuinfo
        cpu_info = cpuinfo.get_cpu_info()["brand_raw"]
        cpu_cores = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)
        ram_info = psutil.virtual_memory()
        ram_total = ram_info.total
        os_info = platform.system()
        os_version = platform.version()
        left_align(f"OS: {os_version} ({os_info})", box_width=width)
        left_align(f"CPU   Model: {cpu_info}", box_width=width)
        left_align(f"CPU   Cores: {cpu_cores}", box_width=width)
        left_align(f"CPU Threads: {cpu_threads}", box_width=width)

        gpu_entries = []
        if os_info == "Windows":
            try:
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "Get-CimInstance Win32_VideoController | "
                     "Select-Object Name, AdapterRAM | "
                     "ForEach-Object { \"$($_.Name)|$($_.AdapterRAM)\" }"],
                    capture_output=True, text=True, timeout=10,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                )
                for line in result.stdout.strip().splitlines():
                    line = line.strip()
                    if not line or "|" not in line:
                        continue
                    name, vram_bytes_str = line.split("|", 1)
                    name = name.strip()
                    try:
                        vram_bytes = int(vram_bytes_str.strip())
                    except (ValueError, TypeError):
                        vram_bytes = 0
                    if not name:
                        continue
                    if vram_bytes > 0:
                        gpu_entries.append((name, vram_bytes / (1024 ** 3)))
                    else:
                        gpu_entries.append((name, 0))
            except Exception:
                pass
        elif os_info == "Linux":
            try:
                result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=10)
                for line in result.stdout.splitlines():
                    lower = line.lower()
                    if "vga" in lower or "3d" in lower or "display" in lower:
                        parts = line.split(": ", 1)
                        name = parts[1].strip() if len(parts) > 1 else line.strip()
                        gpu_entries.append((name, 0))
            except Exception:
                pass
            import glob as globmod
            for i, (name, vram) in enumerate(gpu_entries):
                vram_paths = globmod.glob(f"/sys/class/drm/card{i}-device/mem_info_vram_total")
                if not vram_paths:
                    vram_paths = globmod.glob(f"/sys/class/drm/card{i}/device/mem_info_vram_total")
                for vp in vram_paths:
                    try:
                        with open(vp) as f:
                            vram_bytes = int(f.read().strip())
                        gpu_entries[i] = (name, vram_bytes / (1024 ** 3))
                    except (ValueError, OSError):
                        pass
        elif os_info == "Darwin":
            try:
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True, text=True, timeout=10
                )
                current_name = None
                for line in result.stdout.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("Chipset Model:") or stripped.startswith("Chip Model:"):
                        current_name = stripped.split(":", 1)[1].strip()
                    elif stripped.startswith("VRAM (Total):") and current_name:
                        vram_str = stripped.split(":", 1)[1].strip()
                        gpu_entries.append((current_name, vram_str))
                        current_name = None
                if current_name and current_name not in [e[0] for e in gpu_entries]:
                    gpu_entries.append((current_name, 0))
            except Exception:
                pass

        for name, vram in gpu_entries:
            left_align(f"GPU   Model: {name}", box_width=width)
            if isinstance(vram, (int, float)) and vram > 0:
                left_align(f"GPU    VRAM: {vram:.2f} GB", box_width=width)
            elif isinstance(vram, str):
                left_align(f"GPU    VRAM: {vram}", box_width=width)

        left_align(f"SYSTEM  RAM: {ram_total / (1024 ** 3):.2f} GB", box_width=width)
    except Exception:
        pass


class ReportEngine:

    def __init__(self, parent):
        self._p = parent

    def _get_em(self):
        return getattr(self._p, 'em', self._p)

    def _has_method(self, name):
        return hasattr(self._p, name) and callable(getattr(self._p, name))

    def report_madm(self, all_metrics=False, feloopy_info=True, sys_info=False,
                     model_info=True, sol_info=True, metric_info=True,
                     ideal_pareto=None, ideal_point=None, show_tensors=False,
                     show_detailed_tensors=False, save=None, decimals=4, width=78, style=1,
                     copy_to_clipboard=False):

        p = self._p
        p.show_tensor = show_tensors
        p.show_detailed_tensors = show_detailed_tensors
        p.output_decimals = 4

        box = report(width=width, style=style)
        box._buffered = True

        if feloopy_info:
            self._print_feloopy_info_madm(p, box=box, width=width)

        if sys_info:
            self._print_sys_info_verbose(p, box=box, width=width)

        if model_info:
            self._print_model_info_madm(p, box=box, width=width)

        if sol_info:
            self._print_solve_info_madm(p, box=box, width=width)

        if metric_info:
            self._print_metric_info_madm(p, box=box, width=width)

        self._generate_decision_info(box=box)

        box.render(copy_to_clipboard=copy_to_clipboard)

    def report_search(self, style=1, skip_system_information=True,
                      show_elements=False, width=78, skip=False, full=False,
                      save=None, copy_to_clipboard=False, hidden_variables=False, show_info=False):

        p = self._p

        if getattr(p, 'method', None) == 'sequential':
            return self.report_sequential(
                style=style, skip_system_information=skip_system_information,
                show_elements=show_elements, width=width, save=save,
                copy_to_clipboard=copy_to_clipboard)

        stdout_origin = None
        if save is not None:
            stdout_origin = sys.stdout
            sys.stdout = open(save, "w", encoding="utf-8")

        try:
            box = report(width=width, style=style)
            box._buffered = True

            _method = getattr(p, 'method', None)
            if not getattr(p, 'sensitivity_analyzed', False) and _method not in ('madm', 'sequential'):
                _dataset = getattr(p, 'inputdata', None)
                if _dataset is not None and hasattr(_dataset, 'data') and len(_dataset.data) > 0:
                    from ..helpers.formatter import start_progress, end_progress
                    start_progress(message="Analyzing...", spinner="dots",
                                   color="cyan", show_elapsed=True)
                    try:
                        self._try_auto_sensitivity(p)
                        end_progress(success_message="Analyzed",
                                     show_elapsed=False)
                    except Exception:
                        end_progress(success_message=None,
                                     failure_message="Analysis failed",
                                     success=False, show_elapsed=False)
                        raise

            # Resolve what makes an infeasible model infeasible before any
            # box is rendered: native IIS where the interface has one, the
            # universal progressive batch-removal filter otherwise.  A
            # healthy model never pays for this.
            if _method not in ('madm', 'sequential'):
                try:
                    self._resolve_infeasibility(p)
                except Exception:
                    pass

            self.report_status(style=style, width=width)

            self.report_specs(style=style, width=width, skip_system_information=skip_system_information, box=box)
            self.report_model(style=style, width=width, box=box)
            if full:
                self.report_formulation(style=style, width=width, box=box)
            self.report_metrics(style=style, width=width, box=box)
            self.report_decision(style=style, key_vars=p.key_vars, show_elements=show_elements, width=width, skip=skip, box=box, hidden_variables=hidden_variables)
            self.report_diagnostics(style=style, width=width, box=box)
            self.report_sensitivity(style=style, width=width, box=box)
            self.report_benchmark(style=style, width=width, box=box)
            if show_info:
                self.report_info(style=style, width=width, box=box)

            box.render(copy_to_clipboard=copy_to_clipboard)
        finally:
            if save is not None:
                sys.stdout.close()
                sys.stdout = stdout_origin

    def report_sequential(self, style=1, skip_system_information=True,
                          show_elements=False, width=78, save=None,
                          copy_to_clipboard=False):
        """Compact report for sequential decision making (Powell's SDM)."""
        import datetime
        from colorama import Fore, init as colorama_init
        colorama_init(autoreset=True)

        p = self._p
        result = getattr(p, 'result', None)
        solutions = getattr(p, 'solutions', {}) or {}
        stats = solutions.get('stats', {})
        traces = solutions.get('traces', [])

        stdout_origin = None
        if save is not None:
            stdout_origin = sys.stdout
            sys.stdout = open(save, "w", encoding="utf-8")

        try:
            box = report(width=width, style=style)
            box._buffered = True

            self.report_status(style=style, width=width)

            self.report_specs(style=style, width=width, skip_system_information=skip_system_information, box=box)

            # --- Model ---
            _sdp_obj = getattr(p, '_sdp', None)
            _sdp_policy = getattr(p, '_sdp_policy', None)
            if _sdp_obj is not None:
                _all_vars = getattr(_sdp_obj, '_sdp_vars', {})
                n_state = sum(1 for v in _all_vars.values() if v['type'] == 'state')
                n_dp = sum(1 for v in _all_vars.values() if v['type'] == 'dp')
                n_exo = sum(1 for v in _all_vars.values() if v['type'] == 'exo')
                n_tpar = sum(1 for v in _all_vars.values() if v['type'] == 'tpar')
                box.top(left="Model")
                box.row(left="Type", right="State  Decision  Exog  Theta")
                box.row(left="Count", right=f"{n_state:>5}  {n_dp:>8}  {n_exo:>4}  {n_tpar:>5}")
                box.bottom()

            # --- Metric ---
            mc = np.asarray(solutions.get('total_costs', []))
            dc = np.asarray(solutions.get('discounted_costs', []))
            if len(mc) > 0:
                box.top(left="Metric")
                box.row(left="Metric",
                        right="STG    CFM    OGR    PVR     NZR      MGT     CPT")
                mean_mc = np.mean(mc)
                std_mc = np.std(mc)
                box.row(left="Value",
                        right=f"  -      -      -      -   {std_mc:>5.2f}   "
                              f"{mean_mc:>6.2f}      -")
                box.bottom()

            # --- Optimal Policy ---
            if _sdp_policy is not None and hasattr(_sdp_policy, 'theta'):
                theta = _sdp_policy.theta
                if len(theta) > 0:
                    box.top(left="Optimal Policy")
                    ppar_names = []
                    _tpar_info = getattr(_sdp_policy, '_tpar_info', {})
                    if _tpar_info:
                        ppar_names = list(_tpar_info.keys())
                    else:
                        if _sdp_obj is not None:
                            _all_vars = getattr(_sdp_obj, '_sdp_vars', {})
                            ppar_names = [k for k, v in _all_vars.items()
                                          if v.get('type') == 'tpar']
                    for i, val in enumerate(theta):
                        if i < len(ppar_names):
                            nm = ppar_names[i]
                        else:
                            nm = f'theta[{i}]'
                        if isinstance(val, np.ndarray):
                            val_str = np.array2string(
                                val, precision=6, separator=',',
                                suppress_small=True)
                        else:
                            val_str = f"{val:.6f}"
                        box.row(left=nm, right=val_str)
                    box.bottom()

            # --- Decision ---
            if len(mc) > 0:
                box.top(left="Decision")
                mc_arr = np.asarray(mc)
                box.row(left="OBJ:",
                        right=f"{np.mean(mc_arr):.2f}")
                if _sdp_policy is not None and hasattr(_sdp_policy, 'theta'):
                    theta = _sdp_policy.theta
                    if len(theta) > 0:
                        ppar_names = []
                        _tpar_info = getattr(_sdp_policy, '_tpar_info', {})
                        if _tpar_info:
                            ppar_names = list(_tpar_info.keys())
                        else:
                            if _sdp_obj is not None:
                                _all_vars = getattr(_sdp_obj, '_sdp_vars', {})
                                ppar_names = [k for k, v in _all_vars.items()
                                              if v.get('type') == 'tpar']
                        for i, val in enumerate(theta):
                            if i < len(ppar_names):
                                nm = ppar_names[i]
                            else:
                                nm = f'theta[{i}]'
                            if isinstance(val, np.ndarray):
                                val_str = np.array2string(
                                    val, precision=4, separator=',',
                                    suppress_small=True)
                            else:
                                val_str = f"{val:.4f}"
                            box.row(left=f"{nm} =", right=val_str)
                box.bottom()

            # --- Optimisation Results (if available) ---
            opt_result = solutions.get('opt_result', None)
            if opt_result is not None:
                box.top(left="Optimisation")
                box.row(left="Solver",
                        right=str(getattr(opt_result, 'solver', 'n/a')))
                box.row(left="Evaluations",
                        right=str(getattr(opt_result, 'n_evaluations', 0)))
                init_th = getattr(opt_result, 'init_theta', None)
                best_th = getattr(opt_result, 'best_theta', None)
                if init_th is not None:
                    init_str = np.array2string(
                        np.asarray(init_th), precision=4, separator=',',
                        suppress_small=True)
                    box.row(left="Initial theta", right=init_str)
                if best_th is not None:
                    best_str = np.array2string(
                        np.asarray(best_th), precision=4, separator=',',
                        suppress_small=True)
                    box.row(left="Optimised theta", right=best_str)
                init_s = getattr(opt_result, 'init_stats', {})
                best_s = getattr(opt_result, 'best_stats', {})
                init_cost = init_s.get('mean_cost', 0) if isinstance(
                    init_s, dict) else 0
                best_cost = best_s.get('mean_cost', 0) if isinstance(
                    best_s, dict) else 0
                if init_cost != 0:
                    improvement = (best_cost - init_cost) / init_cost * 100
                else:
                    improvement = 0
                box.row(left="Initial cost",
                        right=f"{init_cost:.4f}")
                box.row(left="Optimised cost",
                        right=f"{best_cost:.4f}")
                box.row(left="Improvement",
                        right=f"{improvement:+.1f}%")
                box.bottom()

            box.render(copy_to_clipboard=copy_to_clipboard)
        finally:
            if save is not None:
                sys.stdout.close()
                sys.stdout = stdout_origin

    def _tensor_row(self, box, p, name, value, detailed):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            p.display_as_tensor(name, value, detailed)
        for line in buf.getvalue().splitlines():
            if not line.strip():
                continue
            if line.startswith('\u2502') and line.endswith('\u2502') and len(line) >= 2:
                inner = line[1:-1]
                if inner.startswith(' '):
                    inner = inner[1:]
                line = inner.rstrip()
            if line:
                box.row(left=line)

    def _generate_decision_info(self, box=None, p=None):
        if box is None:
            box = report(width=78)
        if p is None:
            p = self._p

        box.top(left='Decision')

        if not p.show_tensor:

            if p.features['weights_found']:

                for i in range(p.features['number_of_criteria']):

                    if p.madam_method in ['fuzzy_ahp_method', 'fuzzy_bw_method']:
                        self._tensor_row(box, p, f'fw[{i}]', np.round(p.fuzzy_weights[i],p.output_decimals), p.show_detailed_tensors)
                    else:
                        self._tensor_row(box, p, f'w[{i}]', np.round(p.weights[i],p.output_decimals), p.show_detailed_tensors)

                if p.madam_method in ['fuzzy_bw_method']:
                    for i in range(p.features['number_of_criteria']):
                        self._tensor_row(box, p, f'w[{i}]', np.round(p.weights[i],p.output_decimals), p.show_detailed_tensors)

            if p.features['rpc_found']:
                for i in range(p.features['number_of_criteria']):
                    self._tensor_row(box, p, f'rpc[{i}]', np.round(p.R_plus_C[i],p.output_decimals), p.show_detailed_tensors)

            if p.features['rmc_found']:
                for i in range(p.features['number_of_criteria']):
                    self._tensor_row(box, p, f'rmc[{i}]', np.round(p.R_minus_C[i],p.output_decimals), p.show_detailed_tensors)

            if p.features['ranks_found']:

                for i in range(len(p.ranks)):
                    self._tensor_row(box, p, f'r[{i}]', np.round(p.ranks[i],p.output_decimals), p.show_detailed_tensors)

            if p.features['classification_found']:

                for i in range(p.features['number_of_alternatives']):
                    self._tensor_row(box, p, f'c[{i}]', np.round(p.classification[i],p.output_decimals), p.show_detailed_tensors)

            if p.features['selection_found']:

                for i in range(p.features['number_of_alternatives']):
                    if p.selection[i,2] == 1:
                        self._tensor_row(box, p, f's[{int(p.selection[i,0])}]', np.round(p.selection[i,1],p.output_decimals), p.show_detailed_tensors)

            if p.features['dpr_found']:

                for i in range(p.features['number_of_criteria']):
                    self._tensor_row(box, p, f'dpr[{i}]', np.round(p.D_plus_R[i],p.output_decimals), p.show_detailed_tensors)

            if p.features['dmr_found']:

                for i in range(p.features['number_of_criteria']):
                    self._tensor_row(box, p, f'dmr[{i}]', np.round(p.D_minus_R[i],p.output_decimals), p.show_detailed_tensors)

            if p.features['d_rank_found']:
                for i in range(len(p.rank_D)):
                    self._tensor_row(box, p, f'd_r[{i}]', p.rank_D[i], p.show_detailed_tensors)

            if p.features['a_rank_found']:
                for i in range(len(p.rank_A)):
                    self._tensor_row(box, p, f'a_r[{i}]', p.rank_A[i], p.show_detailed_tensors)

            if p.features['p_rank_found']:
                for i in range(len(p.rank_P)):
                    self._tensor_row(box, p, f'p_r[{i}]', p.rank_P[i], p.show_detailed_tensors)
        else:

            if p.features['weights_found']:

                if p.madam_method in ['fuzzy_ahp_method']:
                    self._tensor_row(box, p, 'fwv', np.round(p.fuzzy_weights,p.output_decimals), p.show_detailed_tensors)
                else:
                    self._tensor_row(box, p, 'wv', np.round(p.weights,p.output_decimals), p.show_detailed_tensors)

            if p.features['ranks_found']:

                if p.madam_method in ['fuzzy_aras_method']:
                    self._tensor_row(box, p, 'frv', np.round(p.fuzzy_ranks,p.output_decimals), p.show_detailed_tensors)
                else:
                    self._tensor_row(box, p, 'rv', np.round(p.ranks,p.output_decimals), p.show_detailed_tensors)

            if p.features['classification_found']:

                self._tensor_row(box, p, f'c', np.round(p.classification,p.output_decimals), p.show_detailed_tensors)

            if p.features['d_rank_found']:
                self._tensor_row(box, p, f'd_rv', p.rank_D, p.show_detailed_tensors)

            if p.features['a_rank_found']:
                self._tensor_row(box, p, f'a_rv', p.rank_A, p.show_detailed_tensors)

            if p.features['p_rank_found']:
                self._tensor_row(box, p, f'p_rv', p.rank_P, p.show_detailed_tensors)

        if p.features['global_concordance_found']:
            self._tensor_row(box, p, 'gcm', np.round(p.global_concordance,p.output_decimals), p.show_detailed_tensors)

        if p.features['concordance_found']:
            self._tensor_row(box, p, 'ccm', np.round(p.concordance,p.output_decimals), p.show_detailed_tensors)

        if p.features['discordance_found']:
            self._tensor_row(box, p, 'dcm', np.round(p.discordance,p.output_decimals), p.show_detailed_tensors)

        if p.features['credibility_found']:
            self._tensor_row(box, p, 'crm', np.round(p.credibility,p.output_decimals), p.show_detailed_tensors)

        if p.features['dominance_found']:
            self._tensor_row(box, p, 'dmm', np.round(p.dominance,p.output_decimals), p.show_detailed_tensors)

        if p.features['kernel_found']:
            self._tensor_row(box, p, 'kernel',  p.kernel, p.show_detailed_tensors)
            
        if p.features['dominated_found']:
            self._tensor_row(box, p, 'dominated', p.dominated, p.show_detailed_tensors)

        if p.features['dominance_s_found']:
            self._tensor_row(box, p, 'dmm_s', p.dominance_s, p.show_detailed_tensors)

        if p.features['dominance_w_found']:
            self._tensor_row(box, p, 'dmm_w', p.dominance_w, p.show_detailed_tensors)

        p.features['dominance_s_found'] = False
        p.features['dominance_w_found'] = False
        p.features['d_rank_found'] = False
        p.features['a_rank_found'] = False
        p.features['n_rank_found'] = False
        p.features['p_rank_found'] = False

        box.bottom()

    def _print_feloopy_info_madm(self, p, box=None, width=78):
        if box is None:
            box = report(width=width)
        import datetime
        now = datetime.datetime.now()
        date_str = now.strftime("Date: %Y-%m-%d")
        time_str = now.strftime("Time: %H:%M:%S")
        from colorama import Fore
        box.top(left=f"FelooPy v{__version__}")
        box.row(left=date_str, right=time_str)
        if 'pydecision' in p.interface_name.lower():
            interface = 'pydecision'
            box.row(left=f"Interface: {interface}", right=f"Solver: {p.madam_method}")
        box.bottom()
        if not box._buffered:
            box.render()

    def _print_sys_info_verbose(self, p, box=None, width=78):
        if box is None:
            box = report(width=width)
        box.top(left="System")
        box.capture(detect_system_info_verbose, width=width)
        box.bottom()
        if not box._buffered:
            box.render()

    def _print_model_info_madm(self, p, box=None, width=78):
        if box is None:
            box = report(width=width)
        box.top(left="Model")
        box.row(left=f"Name: {p.model_name}")
        for i in p.features.keys():
            if 'defined' in i:
                box.row(left=i)
        for i in p.features.keys():
            if 'found' in i and p.features[i]=='True':
                box.row(left=i)
        try:
            box.row(left="Number of criteria:", right=str(p.features['number_of_criteria']))
        except:
            ""
        try:
            box.row(left="Number of alternatives:", right=str(p.features['number_of_alternatives']))
        except:
            ""
        box.bottom()
        if not box._buffered:
            box.render()

    def _print_solve_info_madm(self, p, box=None, width=78):
        if box is None:
            box = report(width=width)
        box.top(left="Solve")
        box.row(left=f"Method: {p.madam_method}")
        box.row(left=f"Status: {p.status}")
        box.bottom()
        if not box._buffered:
            box.render()

    def _print_metric_info_madm(self, p, show_top=True, box=None, width=78):
        if box is None:
            box = report(width=width)
        if show_top:
            box.top(left="Metric")
        if p.features['inconsistency_found']:
            box.row(left='Inconsistency:', right=str(np.round(p.inconsistency,4)))
        import time
        elapsed_time_seconds = p.finish - p.start
        elapsed_time_microseconds = int(elapsed_time_seconds * 1_000_000)
        elapsed_time_formatted = time.strftime('%H:%M:%S', time.gmtime(elapsed_time_seconds))
        if show_top:
            box.row(left='CPT (microseconds):', right=str(elapsed_time_microseconds))
            box.row(left='CPT (hour:min:sec):', right=elapsed_time_formatted)
        if show_top:
            box.bottom()
        if not box._buffered:
            box.render()

    def report_status(self, style=1, width=78):
        from colorama import init, Fore
        init(autoreset=True)
        p = self._p
        phealthy = "Unknown"
        _is_multilevel = getattr(p, 'approach', None) == 'multilevel'
        # a sequential run never solves self.em (it is an untouched
        # placeholder), so health comes from search.healthy() instead
        _em_healthy = (p.healthy() if getattr(p, 'method', None) == 'sequential'
                       else p.em.healthy())
        if _em_healthy or p.method=="madm" or p.sensitivity_analyzed or _is_multilevel or (p.number_of_objectives!=1 and len(p.solutions)!=0):
            phealthy = f"{Fore.GREEN}\u2713 Healthy"
            if p.inputdata:
                if type(p.inputdata)!=dict:
                    p.dataset_size = p.inputdata.size
                    p.big_m_value   = p.inputdata.possible_big_m
                    p.epsilon_value = p.inputdata.possible_epsilon
                    if p.method in ["exact", "heuristic", "convex"]:
                        _solutions = p.solutions
                        _features = p.em.features
                        bvar_unreliable = p.is_value_unreliable(_solutions, [0,1], _features, "bvar")
                        ivar_unreliable = p.is_value_unreliable(_solutions, [0, p.big_m_value], _features, "ivar")
                        pvar_unreliable = p.is_value_unreliable(_solutions, [0, p.big_m_value], _features, "pvar")
                        fvar_unreliable = p.is_value_unreliable(_solutions, [-p.big_m_value, p.big_m_value], _features, "fvar")
                        if bvar_unreliable or ivar_unreliable or pvar_unreliable or fvar_unreliable:
                            phealthy+="\n"+f"{Fore.MAGENTA}\u26a0 Unreliable"

                        bvar_imprecise = p.is_value_impresice(_solutions, [0,1], _features, "bvar")
                        ivar_imprecise = p.is_value_impresice(_solutions, [0, p.big_m_value], _features, "ivar")
                        pvar_imprecise = p.is_value_impresice(_solutions, [0, p.big_m_value], _features, "pvar")
                        fvar_imprecise = p.is_value_impresice(_solutions, [-p.big_m_value, p.big_m_value], _features, "fvar")
                        if bvar_imprecise or ivar_imprecise or pvar_imprecise or fvar_imprecise:
                            phealthy+="\n"+f"{Fore.YELLOW}\u26a0 Imprecise"
        else:
            phealthy = f"{Fore.RED}\u2717 Unhealthy"
        print(phealthy)

    def report_specs(self, style=1, width=78, skip_system_information=False, box=None):
        p = self._p
        if box is None:
            box = report(width=width, style=style)
        import datetime
        current_datetime = datetime.datetime.now()
        formatted_date = current_datetime.strftime("%Y-%m-%d")
        formatted_time = current_datetime.strftime("%H:%M:%S")
        from colorama import Fore
        box.top(left=f"FelooPy v{__version__}", right=f"Released {__release_month__} {__release_year__}")
        box.clear_columns(list_of_strings=["", f"Interface: {p.interface}"], label=f"Date: {formatted_date}", max_space_between_elements=4)
        configured = False
        try:
            if len(p.em.features.get('solver_options', {})) >= 1:
                configured = True
        except:
            pass
        solver_display = f"Solver: {p.solver}{'*' if configured else ''}"
        box.clear_columns(list_of_strings=["", solver_display], label=f"Time: {formatted_time}", max_space_between_elements=4)
        box.clear_columns(list_of_strings=["",f"Method: {p.method}"], label= f"Name: {p.name}", max_space_between_elements=4)
        _is_multilevel = getattr(p, 'approach', None) == 'multilevel'
        if p.method in ["exact", "convex", "constraint", "uncertain","heuristic"]:
            if _is_multilevel:
                ptype = "ML"
            elif hasattr(p, 'approach') and p.approach == 'bilevel':
                ptype = "BL"
            elif p.number_of_objectives == 1:
                ptype = "SO"
            elif p.number_of_objectives == 2:
                ptype = "BO"
            else:
                ptype = "MO"
            if p.method == "constraint":
                ptype = f"{ptype} CP"
            elif p.method == "heuristic":
                ptype = f"{ptype} GPP"
            elif p.method == "uncertain":
                ptype = f"{ptype} U"
            elif p.method == "convex":
                ptype = f"{ptype} CVX"
        else:
            ptype = f"MA {p.method.upper()}" if p.method else "MA"
        try:
            prog_type = p.em.features.get('problem_type', None)
            if prog_type:
                if p.em.features.get('_auto_lin_active'):
                    orig_type = p.em.features.get('original_problem_type', '?')
                    ptype = f"{ptype} {prog_type} (auto)"
                else:
                    ptype = f"{ptype} {prog_type}"
        except Exception:
            pass

        if skip_system_information is False:
            try:
                box.bottom(right=detect_system_info())
            except:
                box.bottom()
        else:
            box.bottom()

        if not box._buffered:
            box.render()

    def report_model(self, style=1, width=78, box=None):
        p = self._p
        em = p.em
        if box is None:
            box = report(width=width, style=style)
        if p.method != 'madm':
            box.top(left="Model")

            def _build_table(raw_class, raw_size, headers_list):
                visible = [(h, c, s) for h, c, s in zip(headers_list, raw_class, raw_size) if c != 0]
                if not visible:
                    visible = list(zip(headers_list, raw_class, raw_size))
                hdrs = [v[0] for v in visible]
                class_vals = [str(v[1]) for v in visible]
                size_vals = [str(v[2]) for v in visible]
                all_rows = [hdrs, class_vals, size_vals]
                col_widths = [max(len(row[i]) for row in all_rows) for i in range(len(hdrs))]
                hdr_str = "  ".join(v.rjust(col_widths[i]) for i, v in enumerate(hdrs))
                cls_str = "  ".join(v.rjust(col_widths[i]) for i, v in enumerate(class_vals))
                sz_str = "  ".join(v.rjust(col_widths[i]) for i, v in enumerate(size_vals))
                return hdr_str, cls_str, sz_str

            headers = ["B", "I", "P", "F", "E", "S", "O", "C"]

            def _size(key, fallback=0):
                """Total element size of a feature.

                Accepts either a scalar feature (e.g. ``_original_bvar_size``)
                or a ``[count, size]`` counter (e.g. ``total_variable_counter``).
                """
                try:
                    val = em.features.get(key, fallback)
                    if isinstance(val, (list, tuple)):
                        return int(val[1]) if len(val) > 1 else int(fallback)
                    return int(val)
                except Exception:
                    return int(fallback)

            def _dims(n_con, n_var):
                """``(constraints x variables)`` suffix for the program type."""
                return f" ({n_con} x {n_var})"

            if em.features.get('_auto_lin_active') and '_original_variable_count' in em.features:
                orig_type = em.features.get('original_problem_type', '?')
                al_type = em.features.get('problem_type', '?')
                try:
                    _nobj = p.number_of_objectives if hasattr(p, 'number_of_objectives') else 0
                    if _nobj == 1:
                        _obj_prefix = "SO"
                    elif _nobj == 2:
                        _obj_prefix = "BO"
                    elif _nobj > 2:
                        _obj_prefix = "MO"
                    else:
                        _obj_prefix = ""
                except:
                    _obj_prefix = ""
                _orig_has_con = em.features.get('_original_con_count', 0) > 0
                _orig_con = "C" if _orig_has_con else "UC"
                _al_has_con = em.features.get("constraint_counter", [0, 0])[0] > 0
                _al_con = "C" if _al_has_con else "UC"
                _orig_prefixed = f"{_orig_con} {_obj_prefix} {orig_type}" if _obj_prefix else f"{_orig_con} {orig_type}"
                _al_prefixed = f"{_al_con} {_obj_prefix} {al_type}" if _obj_prefix else f"{_al_con} {al_type}"

                # Original totals: snapshotted per-type sizes (auto-linearization
                # never creates event/sequential vars, so current values are the
                # original ones). Falls back to the live counters if the snapshot
                # is missing.
                _orig_con_total = _size('_original_con_size', _size('_original_con_count'))
                _orig_var_total = (
                    _size('_original_bvar_size', _size('binary_variable_counter'))
                    + _size('_original_ivar_size', _size('integer_variable_counter'))
                    + _size('_original_pvar_size', _size('positive_variable_counter'))
                    + _size('_original_fvar_size', _size('free_variable_counter'))
                    + _size('event_variable_counter')
                    + _size('sequential_variable_counter')
                )
                _orig_prefixed += _dims(_orig_con_total, _orig_var_total)
                _al_prefixed += _dims(
                    _size('constraint_counter'),
                    _size('total_variable_counter'),
                )

                orig_raw_class = [
                    em.features.get('_original_bvar_count', 0),
                    em.features.get('_original_ivar_count', 0),
                    em.features.get('_original_pvar_count', 0),
                    em.features.get('_original_fvar_count', 0),
                    em.features.get('event_variable_counter', [0, 0])[0],
                    em.features.get('sequential_variable_counter', [0, 0])[0],
                    em.features.get('_original_obj_count', 0),
                    em.features.get('_original_con_count', 0),
                ]
                orig_raw_size = [
                    em.features.get('_original_bvar_size', 0),
                    em.features.get('_original_ivar_size', 0),
                    em.features.get('_original_pvar_size', 0),
                    em.features.get('_original_fvar_size', 0),
                    em.features.get('event_variable_counter', [0, 0])[1],
                    em.features.get('sequential_variable_counter', [0, 0])[1],
                    em.features.get('_original_obj_count', 0),
                    em.features.get('_original_con_size', 0),
                ]

                al_raw_class = [
                    em.features.get("binary_variable_counter", [0, 0])[0],
                    em.features.get("integer_variable_counter", [0, 0])[0],
                    em.features.get("positive_variable_counter", [0, 0])[0],
                    em.features.get("free_variable_counter", [0, 0])[0],
                    em.features.get("event_variable_counter", [0, 0])[0],
                    em.features.get("sequential_variable_counter", [0, 0])[0],
                    em.features.get("objective_counter", [0, 0])[0],
                    em.features.get("constraint_counter", [0, 0])[1],
                ]
                al_raw_size = [
                    em.features.get("binary_variable_counter", [0, 0])[1],
                    em.features.get("integer_variable_counter", [0, 0])[1],
                    em.features.get("positive_variable_counter", [0, 0])[1],
                    em.features.get("free_variable_counter", [0, 0])[1],
                    em.features.get("event_variable_counter", [0, 0])[1],
                    em.features.get("sequential_variable_counter", [0, 0])[1],
                    em.features.get("objective_counter", [0, 0])[1],
                    em.features.get("constraint_counter", [0, 0])[1],
                ]

                def _diff_suffix(o, a):
                    if o == a:
                        return ""
                    d = a - o
                    return f"(+{d})" if d > 0 else f"({d})"

                def _fmt_with_diff(val, suffix):
                    if not suffix:
                        return str(val)
                    return str(val) + " " + suffix

                def _build_table_with_suffs(raw_class, raw_size, headers_list, suff_class, suff_size):
                    display_c = [_fmt_with_diff(raw_class[i], suff_class[i]) for i in range(len(headers_list))]
                    display_s = [_fmt_with_diff(raw_size[i], suff_size[i]) for i in range(len(headers_list))]
                    visible = [(headers_list[i], display_c[i], display_s[i]) for i in range(len(headers_list)) if raw_class[i] != 0 or raw_size[i] != 0]
                    if not visible:
                        visible = [(headers_list[i], display_c[i], display_s[i]) for i in range(len(headers_list))]
                    hdrs = [v[0] for v in visible]
                    cls = [v[1] for v in visible]
                    szs = [v[2] for v in visible]
                    all_rows = [hdrs, cls, szs]
                    col_widths = [max(len(row[i]) for row in all_rows) for i in range(len(hdrs))]
                    hdr_str = "  ".join(hdrs[i].rjust(col_widths[i]) for i in range(len(hdrs)))
                    cls_str = "  ".join(cls[i].rjust(col_widths[i]) for i in range(len(hdrs)))
                    sz_str = "  ".join(szs[i].rjust(col_widths[i]) for i in range(len(hdrs)))
                    return hdr_str, cls_str, sz_str

                oh, oc, os = _build_table(orig_raw_class, orig_raw_size, headers)
                lbl = "Original Program" + "  "
                pad = width - 4 - len(lbl) - len(_orig_prefixed)
                inner = lbl + " " * max(pad, 1) + _orig_prefixed
                box._emit(f"{box._border} {inner} {box._border}", "row")
                for label, content in [("Element", oh), ("Count", oc), ("Size", os)]:
                    lbl = f"{label}" + "  "
                    pad = width - 4 - len(lbl) - len(content)
                    inner = lbl + " " * max(pad, 1) + content
                    box._emit(f"{box._border} {inner} {box._border}", "row")

                diff_class_list = [_diff_suffix(oc_, ac_) for oc_, ac_ in zip(orig_raw_class, al_raw_class)]
                diff_size_list = [_diff_suffix(os_, as_) for os_, as_ in zip(orig_raw_size, al_raw_size)]

                ah, ac, as_ = _build_table_with_suffs(al_raw_class, al_raw_size, headers, diff_class_list, diff_size_list)
                lbl = "Linearized Program" + "  "
                pad = width - 4 - len(lbl) - len(_al_prefixed)
                inner = lbl + " " * max(pad, 1) + _al_prefixed
                box._emit(f"{box._border} {inner} {box._border}", "row")
                for label, content in [("Element", ah), ("Count", ac), ("Size", as_)]:
                    lbl = f"{label}" + "  "
                    pad = width - 4 - len(lbl) - len(content)
                    inner = lbl + " " * max(pad, 1) + content
                    box._emit(f"{box._border} {inner} {box._border}", "row")
            else:
                try:
                    _nobj = p.number_of_objectives if hasattr(p, 'number_of_objectives') else 0
                    if _nobj == 1:
                        _obj_prefix = "SO"
                    elif _nobj == 2:
                        _obj_prefix = "BO"
                    elif _nobj > 2:
                        _obj_prefix = "MO"
                    else:
                        _obj_prefix = ""
                except:
                    _obj_prefix = ""
                _ptype = em.features.get('problem_type', '?')
                if _ptype == '?' and getattr(em, 'solution_method', em.features.get('solution_method', '')) == 'heuristic':
                    _ptype = 'GPP'
                _full_type = f"{_obj_prefix} {_ptype}" if _obj_prefix else _ptype
                _has_con = em.features.get("constraint_counter", [0, 0])[0] > 0
                _con_prefix = "C " if _has_con else "UC "
                _full_type = f"{_con_prefix}{_full_type}"
                _full_type += _dims(
                    _size('constraint_counter'),
                    _size('total_variable_counter'),
                )
                lbl = "Program" + "  "
                pad = width - 4 - len(lbl) - len(_full_type)
                inner = lbl + " " * max(pad, 1) + _full_type
                box._emit(f"{box._border} {inner} {box._border}", "row")

                raw_class = [
                    em.features.get("binary_variable_counter", [0, 0])[0],
                    em.features.get("integer_variable_counter", [0, 0])[0],
                    em.features.get("positive_variable_counter", [0, 0])[0],
                    em.features.get("free_variable_counter", [0, 0])[0],
                    em.features.get("event_variable_counter", [0, 0])[0],
                    em.features.get("sequential_variable_counter", [0, 0])[0],
                    em.features.get("objective_counter", [0, 0])[0],
                    em.features.get("constraint_counter", [0, 0])[0],
                ]
                raw_size = [
                    em.features.get("binary_variable_counter", [0, 0])[1],
                    em.features.get("integer_variable_counter", [0, 0])[1],
                    em.features.get("positive_variable_counter", [0, 0])[1],
                    em.features.get("free_variable_counter", [0, 0])[1],
                    em.features.get("event_variable_counter", [0, 0])[1],
                    em.features.get("sequential_variable_counter", [0, 0])[1],
                    em.features.get("objective_counter", [0, 0])[1],
                    em.features.get("constraint_counter", [0, 0])[1],
                ]
                hdr_str, cls_str, sz_str = _build_table(raw_class, raw_size, headers)
                for label, content in [("Element", hdr_str), ("Count", cls_str), ("Size", sz_str)]:
                    lbl = f"{label}" + "  "
                    pad = width - 4 - len(lbl) - len(content)
                    inner = lbl + " " * max(pad, 1) + content
                    box._emit(f"{box._border} {inner} {box._border}", "row")

            box.bottom()

        if not box._buffered:
            box.render()

    def report_formulation(self, style=1, width=78, box=None):
        import textwrap
        p = self._p
        if box is None:
            box = report(width=width, style=style)
        box.top(left="Formulation")
        box.empty()
        obdirs = 0
        for objective in p.em.features['objectives']:
            if obdirs > len(p.directions)-1:
                box.empty()
            wrapped_objective = textwrap.fill(str(objective), width=width)
            box.print_value("obj", f"{p.em.features['directions'][obdirs]} {wrapped_objective}", sign=":")
            obdirs += 1
        box.empty()
        if  p.em.features['constraint_labels'] and p.em.features['constraint_labels'][0] != None:
            for constraint in sorted(
                zip(p.em.features['constraint_labels'], p.em.features['constraints']),
                key=lambda x: x[0] if x[0] is not None else ''
            ):
                wrapped_constraint = textwrap.fill(str(constraint[1]), width=width)
                box.print_value(str(f"con {constraint[0]}"), str(wrapped_constraint), sign=":")
        else:
            counter = 0
            for constraint in p.em.features['constraints']:
                wrapped_constraint = textwrap.fill(str(constraint), width=78)
                boxed(str(f"con {counter}: {wrapped_constraint}"))
                counter += 1
        box.empty()
        box.bottom()

        if not box._buffered:
            box.render()

    def _collect_lp_analysis(self):
        """Dual/slack and bound facts for the LP analysis section.

        Returns ``(meaningful, n_total_cons, at_lower, at_upper)``, or ``None``
        when the model is not a plain LP or has nothing worth reporting.
        """
        p = self._p
        _is_multilevel = getattr(p, 'approach', None) == 'multilevel'
        if _is_multilevel:
            return None
        if p.method == "heuristic":
            return None
        _has_bvar = p.em.features.get('binary_variable_counter', [0, 0])[0] > 0
        _has_ivar = p.em.features.get('integer_variable_counter', [0, 0])[0] > 0
        if _has_bvar or _has_ivar:
            return None

        # ── Constraint analysis (duals) ──
        meaningful = []
        n_total_cons = 0
        if p.em.features.get('constraint_labels'):
            try:
                _stderr_fd = sys.stderr.fileno()
                _saved_stderr = os.dup(_stderr_fd)
                _devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(_devnull, _stderr_fd)
                os.close(_devnull)
                try:
                    p.em.get_dual(p.em.features['constraint_labels'][0], tensor=False)
                finally:
                    os.dup2(_saved_stderr, _stderr_fd)
                    os.close(_saved_stderr)
            except Exception:
                pass

            slack_data = {}
            for i in p.em.features['constraint_labels']:
                if i is None or not isinstance(i, str) or i.startswith('_'):
                    continue
                try:
                    val = p.em.get_slack(i, tensor=False)
                    if val is not None:
                        slack_data[i] = val
                except Exception:
                    pass

            dual_data = {}
            try:
                _stderr_fd = sys.stderr.fileno()
                _saved_stderr = os.dup(_stderr_fd)
                _devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(_devnull, _stderr_fd)
                os.close(_devnull)
                try:
                    for i in p.em.features['constraint_labels']:
                        if i is None or not isinstance(i, str) or i.startswith('_'):
                            continue
                        try:
                            val = p.em.get_dual(i, tensor=False)
                            if val is not None:
                                dual_data[i] = val
                        except Exception:
                            pass
                finally:
                    os.dup2(_saved_stderr, _stderr_fd)
                    os.close(_saved_stderr)
            except Exception:
                pass

            all_labels = list(dict.fromkeys(list(slack_data.keys()) + list(dual_data.keys())))
            n_total_cons = len(all_labels)
            for label in all_labels:
                s = slack_data.get(label)
                d = dual_data.get(label)
                d_val = float(d) if d is not None else 0.0
                if abs(d_val) > 1e-6:
                    meaningful.append((label, s, d))

        # ── Variable bound analysis ──
        _var_bounds = p.em.features.get('variable_bound', {})
        _solutions = getattr(p, 'solutions', None)
        _solutions = _solutions if isinstance(_solutions, dict) else {}
        _at_lower = []
        _at_upper = []
        for vname, bounds in _var_bounds.items():
            if vname.startswith('_') or 'autolin' in vname or 'sos2' in vname or 'rdiv' in vname:
                continue
            if vname not in _solutions:
                continue
            sol = _solutions[vname]
            lo = float(bounds[0]) if bounds[0] is not None else None
            hi = float(bounds[1]) if bounds[1] is not None else None
            try:
                arr = np.asarray(sol, dtype=float)
            except Exception:
                continue
            if arr.ndim == 0:
                v = float(arr)
                if lo is not None and abs(v - lo) < 1e-6 and abs(lo) > 1e-10:
                    _at_lower.append((vname, v))
                elif hi is not None and abs(v - hi) < 1e-6 and abs(hi) > 1e-10:
                    _at_upper.append((vname, v))
            else:
                for idx in np.ndindex(arr.shape):
                    val = float(arr[idx])
                    label = f"{vname}[{','.join(str(i) for i in idx)}]" if arr.ndim > 1 else f"{vname}[{idx[0]}]"
                    if lo is not None and abs(val - lo) < 1e-6 and abs(lo) > 1e-10:
                        _at_lower.append((label, val))
                    elif hi is not None and abs(val - hi) < 1e-6 and abs(hi) > 1e-10:
                        _at_upper.append((label, val))

        if not meaningful and not _at_lower and not _at_upper:
            return None
        return meaningful, n_total_cons, _at_lower, _at_upper

    @staticmethod
    def _emit_row(box, inner_w, text, kind="row"):
        """Emit one body row padded (and clipped the way ``box.row`` clips)
        to the content width, so the border stays exactly ``width`` wide."""
        text = box._clip(text, inner_w)
        _len = box._visible_len(text)
        if _len < inner_w:
            text = text + " " * (inner_w - _len)
        box._emit(f"{box._border} {text} {box._border}", kind)

    def _emit_lp_analysis_rows(self, box, data, inner_w):
        """Render the dual/slack and bound rows of a collected LP analysis."""
        meaningful, n_total_cons, _at_lower, _at_upper = data
        _val_w = 14
        _label_w = inner_w - _val_w

        def _var_row(lbl, val_str):
            return (lbl[:_label_w].ljust(_label_w)
                    + val_str[:_val_w].rjust(_val_w))

        if meaningful:
            n_nz_dual = len(meaningful)
            self._emit_row(
                box, inner_w,
                f"Constraints ({n_nz_dual}/{n_total_cons} nonzero dual)")
            _cons_label_w = inner_w - _val_w * 2 - 2
            for label, s_val, d_val in meaningful:
                s_str = format_string(s_val) if s_val is not None else "-"
                d_str = format_string(d_val) if d_val is not None else "-"
                l = label[:_cons_label_w].ljust(_cons_label_w)
                s = f"S={s_str}".rjust(_val_w)
                d = f"D={d_str}".rjust(_val_w)
                self._emit_row(box, inner_w, l + s + "  " + d)

        if _at_lower or _at_upper:
            _n_vars = len(_at_lower) + len(_at_upper)
            self._emit_row(box, inner_w, f"Variables at bounds ({_n_vars})")
            for vname, val in _at_lower:
                self._emit_row(box, inner_w,
                               _var_row(f"  {vname}",
                                        f"= LB ({format_string(val)})"))
            for vname, val in _at_upper:
                self._emit_row(box, inner_w,
                               _var_row(f"  {vname}",
                                        f"= UB ({format_string(val)})"))

    def report_lp_insights(self, style=1, width=78, box=None):
        """Standalone LP analysis box.

        The same rows normally render as a section of ``report_diagnostics``.
        """
        data = self._collect_lp_analysis()
        if data is None:
            return
        standalone = box is None
        if standalone:
            box = report(width=width, style=style)
        box.top(left="Analysis")
        self._emit_lp_analysis_rows(box, data, width - 4)
        box.bottom()
        if standalone:
            box.render()

    def _iis_by_batch_removal(self, p):
        """Universal IIS fallback: progressive constraint-batch removal.

        A deletion filter at batch granularity -- drop one batch at a time and
        re-solve; a batch is dropped only when the model is still infeasible
        without it, so what survives is an irreducible set.  Because the whole
        procedure only re-solves through FelooPy it works for every interface,
        including those with no native IIS.  Returns display lines, or
        ``None`` when no batch explains the infeasibility.
        """
        import copy as _copy
        from ..helpers.formatter import (suppress_output, start_progress,
                                         end_progress, update_progress)

        try:
            # only the problem's own constraints: FelooPy's injected rows
            # ('_'-prefixed) are never what the user needs blamed
            _registry = self._problem_batches(
                getattr(p, 'em', p).features.get('constraint_batches'))
        except Exception:
            _registry = {}
        _names = list(_registry)
        _env = getattr(p, 'environment', None)
        if not _names or _env is None or not hasattr(p, 'create_env'):
            return None

        _MISSING = object()
        _by_ref = ('em', 'cpt', 'mgt', '_exclude_batches',
                   '_benders_result', '_cg_result', '_lagrangian_result',
                   '_branching_result')
        _by_copy = ('objective_values', 'solutions', 'data')
        _snap = {a: getattr(p, a, _MISSING) for a in _by_ref}
        for _a in _by_copy:
            _v = getattr(p, _a, _MISSING)
            if _v is _MISSING:
                _snap[_a] = _MISSING
                continue
            try:
                _snap[_a] = _copy.deepcopy(_v)
            except Exception:
                _snap[_a] = _v

        def _restore():
            for _a, _v in _snap.items():
                if _v is _MISSING:
                    try:
                        delattr(p, _a)
                    except AttributeError:
                        pass
                else:
                    setattr(p, _a, _v)

        _boost = ('_benders_result', '_cg_result', '_lagrangian_result',
                  '_branching_result')

        # kept = batches still present in the model under test
        _kept = list(_names)
        _decided = False
        start_progress(message="Isolating conflict...", spinner="dots",
                       color="cyan", show_elapsed=True)
        try:
            for _b in _names:
                if _b not in _kept:
                    continue
                _trial = [x for x in _kept if x != _b]
                update_progress(f"Isolating conflict {_b}")
                try:
                    p._exclude_batches = tuple(x for x in _names
                                               if x not in _trial)
                    with suppress_output():
                        for _a in _boost:
                            try:
                                delattr(p, _a)
                            except AttributeError:
                                pass
                        p.objective_values = None
                        p.create_env(_env, verbose=True)
                        p.run(verbose=True)
                    _healthy = p.em.healthy()
                except Exception:
                    _healthy = None
                if _healthy is True or _healthy is False:
                    _decided = True
                    if _healthy is False:
                        # still infeasible without _b -> _b is not implicated
                        _kept = _trial
        finally:
            end_progress(success_message="Isolated", show_elapsed=False)
            _restore()

        if not _decided or not _kept:
            return None

        _labels = []
        for _b in _kept:
            for _e in (_registry.get(_b, {}).get('elements') or ()):
                _labels.append(_e[1] if isinstance(_e, (tuple, list))
                               and len(_e) > 1 else str(_e))
        if _labels and len(_labels) <= 8:
            return [f"con: {x}" for x in _labels]
        return [f"batch: {_b} ({len(_registry.get(_b, {}).get('elements') or ())})"
                for _b in _kept]

    def _resolve_infeasibility(self, p):
        """Constraint labels behind an infeasible model.

        Returns ``None`` when the question does not apply (healthy model,
        heuristic or multi-objective runs), a list of ``con:`` / ``batch:``
        lines when it does, and an empty list when nothing could be
        identified.  The interface's own IIS is used when it has one;
        everything else falls back to the universal progressive batch-removal
        filter.  Nothing is ever recomputed on a healthy model.
        """
        if p.method == 'heuristic' or p.number_of_objectives != 1:
            return None
        try:
            if p.healthy() is not False:
                return None
        except Exception:
            return None
        if getattr(p, '_diagnostics_iis', None) is not None:
            return p._diagnostics_iis

        _items = []
        try:
            em = getattr(p, 'em', p)
            _native = None
            try:
                _native = em.features.get('debug_info', None) or None
            except Exception:
                _native = None
            if not _native:
                try:
                    _native = em.find_iis()
                except Exception:
                    _native = None
            if _native:
                _items = [str(x) if str(x).startswith('con:') else f"con: {x}"
                          for x in _native]
            else:
                _items = self._iis_by_batch_removal(p) or []
        except Exception:
            _items = []
        p._diagnostics_iis = _items
        return _items

    def report_debug(self, style=1, width=78, box=None):
        """Standalone infeasibility box (normally part of Diagnostics)."""
        p = self._p
        _items = self._resolve_infeasibility(p)
        if _items is None:
            return
        if box is None:
            box = report(width=width, style=style)
        box.top(left="Debug")
        inner_w = width - 4
        if _items:
            self._emit_row(box, inner_w, "Infeasibility is caused by:")
            for _s in _items:
                self._emit_row(box, inner_w, f"  {_s}")
        else:
            self._emit_row(box, inner_w, "Could not identify critical constraints.")
        box.bottom()
        if not box._buffered:
            box.render()

    def report_diagnostics(self, style=1, width=78, box=None):
        """Constraint-batch impact, LP analysis and infeasibility in one box.

        Rendered directly after the Decision box by ``report_search`` and
        silent when there is nothing to diagnose.
        """
        p = self._p

        _batch_rows = getattr(p, 'sensitivity_batch_data', None) or []
        _lp = self._collect_lp_analysis()
        if getattr(p, '_diagnostics_iis', None) is None:
            try:
                _iis = self._resolve_infeasibility(p)
            except Exception:
                _iis = None
        else:
            _iis = p._diagnostics_iis

        if not _batch_rows and _lp is None and _iis is None:
            return

        standalone = box is None
        if standalone:
            box = report(width=width, style=style)
        inner_w = width - 4

        def _fmt_pct(_v):
            return "-" if _v is None else f"{_v:>+8.2f}%"

        def _fmt_pct_w(_v, _w):
            """Percent cell that never grows past ``_w`` columns."""
            if _v is None:
                return "-"
            for _fs in (f"{{:+{_w - 1}.2f}}%", f"{{:+{_w - 1}.1f}}%",
                        f"{{:+{_w - 1}.0f}}%"):
                _s = _fs.format(_v)
                if len(_s) <= _w:
                    return _s
            return _s[:_w]

        def _fmt_obj(_v):
            return "-" if _v is None else f"{_v:>10.3f}"

        def _fmt_time(_v):
            return "-" if _v is None else f"{_v:>8.3f}s"

        box.top(left="Diagnostics")

        if _iis is not None:
            self._emit_row(box, inner_w, "Infeasibility")
            if _iis:
                for _s in _iis:
                    self._emit_row(box, inner_w, f"  {_s}")
            else:
                self._emit_row(box, inner_w,
                               "  Could not identify critical constraints.")

        if _batch_rows:
            self._emit_row(box, inner_w, "Constraint Batches")
            _n_obj = int(getattr(p, 'number_of_objectives', 1) or 1)
            if _n_obj <= 1:
                self._emit_row(
                    box, inner_w,
                    "  " + f"{'Batch':<18} {'N':>3} {'Obj(-B)':>10} "
                           f"{'dObj%':>9} {'Time(-B)':>9} {'dT%':>9} {'Status':<7}")
                for _br in _batch_rows:
                    _bn = str(_br.get('name', ''))
                    if len(_bn) > 18:
                        _bn = _bn[:15] + "..."
                    self._emit_row(
                        box, inner_w,
                        "  " + f"{_bn:<18} {_br.get('n', 0):>3} "
                               f"{_fmt_obj(_br.get('obj_without')):>10} "
                               f"{_fmt_pct(_br.get('obj_pct')):>9} "
                               f"{_fmt_time(_br.get('time_without')):>9} "
                               f"{_fmt_pct(_br.get('time_pct')):>9} "
                               f"{_br.get('status', '?'):<7}")
            else:
                # A multi-objective run has one average per objective, so
                # each objective gets its own change column instead of the
                # single blended 'dObj%'.  'Avg(-B)' (the front-wide
                # average) stays in only while every objective column still
                # fits; the batch name and status columns shrink first.
                _pw = 8
                _sw = 7
                _bw = min(18, 29 - 9 * _n_obj)
                _keep_avg = _bw >= 8

                def _row_w(_bw_, _pw_, _avg):
                    # lead 2, Batch, N, [Avg(-B)], M pct, Time, dT, Status,
                    # plus one space between every pair of the 8-ish fields
                    _nf = 2 + (1 if _avg else 0) + _n_obj + 3
                    return (2 + _bw_ + 3 + (10 if _avg else 0)
                            + _n_obj * _pw_ + 9 + 9 + _sw + (_nf - 1))

                if not _keep_avg:
                    _bw = 18
                while _bw > 4 and _row_w(_bw, _pw, _keep_avg) > inner_w:
                    _bw -= 1
                if _keep_avg and _row_w(_bw, _pw, True) > inner_w:
                    _keep_avg = False
                    _bw = 18
                    while _bw > 4 and _row_w(_bw, _pw, False) > inner_w:
                        _bw -= 1
                while _pw > 6 and _row_w(_bw, _pw, _keep_avg) > inner_w:
                    _pw -= 1

                def _fit(_s, _w):
                    return _s if len(_s) <= _w else _s[:_w]

                _hdr = "  " + f"{_fit('Batch', _bw):<{_bw}} {'N':>3}"
                if _keep_avg:
                    _hdr += " " + f"{'Avg(-B)':>10}"
                for _k in range(_n_obj):
                    _hdr += " " + f"{('dObj%d%%' % (_k + 1)):>{_pw}}"
                _hdr += (f" {'Time(-B)':>9} {'dT%':>9}"
                         f" {_fit('Status', _sw):<{_sw}}")
                self._emit_row(box, inner_w, _hdr)
                for _br in _batch_rows:
                    _bn = _fit(str(_br.get('name', '')), _bw)
                    _row = "  " + f"{_bn:<{_bw}} {_br.get('n', 0):>3}"
                    if _keep_avg:
                        _row += " " + f"{_fmt_obj(_br.get('obj_without')):>10}"
                    _pct_k = _br.get('obj_pct_k')
                    if not isinstance(_pct_k, list):
                        _pct_k = []
                    for _k in range(_n_obj):
                        _v = _pct_k[_k] if _k < len(_pct_k) else None
                        _row += " " + f"{_fmt_pct_w(_v, _pw):>{_pw}}"
                    _row += (f" {_fmt_time(_br.get('time_without')):>9}"
                             f" {_fmt_pct(_br.get('time_pct')):>9}"
                             f" {_fit(str(_br.get('status', '?')), _sw):<{_sw}}")
                    self._emit_row(box, inner_w, _row)

        if _lp is not None:
            self._emit_row(box, inner_w, "LP Analysis")
            self._emit_lp_analysis_rows(box, _lp, inner_w)

        box.bottom()
        if standalone:
            box.render()

    def report_decision(self, style=1, key_vars=[], show_elements=False, width=78, skip=False, box=None, hidden_variables=False):
        p = self._p
        p.key_vars = key_vars
        if box is None:
            box = report(width=width, style=style)

        # Map variable name -> vtype ("bvar"/"btvar"/...) so the decision
        # renderer can decode route/assignment structure only for binary
        # variables (a 0/1 schedule of continuous values is neither).
        try:
            _vars = p.em.features.get('variables', {})
            box.var_types = {
                k[1]: k[0] for k in _vars
                if isinstance(k, tuple) and len(k) == 2 and isinstance(k[1], str)
            }
        except Exception:
            box.var_types = {}

        _is_multilevel = getattr(p, 'approach', None) == 'multilevel'

        if p.method != 'madm':
            box.top(left="Decision")

            if not p.healthy() and p.method != "heuristic" and not _is_multilevel:
                box.row(left="Indecisive")
                box.bottom()
            else:
                try:
                    _is_cp = p.em.features.get('interface_name', '') in ['ortools_cp', 'cplex_cp', 'picat']
                    _use_cp_display = (p.method == "constraint" or _is_cp)
                    if _use_cp_display:
                        import io, contextlib
                        var_buf = io.StringIO()
                        try:
                            with contextlib.redirect_stdout(var_buf):
                                # ``status`` is only set by some solve paths
                                # (a failed get_status() leaves it unset), and
                                # the CP printers can raise: never let that
                                # empty the Decision box.
                                p.em.decision_information_print(
                                    getattr(p.em, 'status', None), False, False)
                        except Exception:
                            pass
                        _var_lines = [
                            ln.strip() for ln in var_buf.getvalue().split("\n")
                            if ln.strip()
                        ]
                        if _var_lines:
                            try:
                                _dirs = getattr(p, 'directions', None)
                                colored = _colorize_obj_values(p.objective_values[0], _dirs, p.objective_values, 0)
                                box.row(left='OBJ: ' + ' '.join(colored))
                            except Exception:
                                pass
                            for line in _var_lines:
                                box.row(left=line)
                            # terms (m.term) are not native variables, so the
                            # CP printer never emits them: add them from the
                            # reported solutions.
                            try:
                                if isinstance(p.solutions, dict):
                                    _tf = p.em.features or {}
                                    _term_names = []
                                    for _src in ('_term_order', '_terms', '_deferred_terms'):
                                        for _n in (_tf.get(_src) or {}):
                                            if _n not in _term_names:
                                                _term_names.append(_n)
                                    _printed = {
                                        ln.split('=')[0].split('[')[0].strip()
                                        for ln in _var_lines
                                    }
                                    for _n in _term_names:
                                        if _n in p.solutions and _n not in _printed:
                                            box.print_element(_n, p.solutions[_n])
                            except Exception:
                                pass
                            box.bottom()
                            return
                        # The CP printer yielded nothing (the native model only
                        # holds one Pareto point, zero-valued variables are
                        # skipped, terms are not printed, ...): fall through to
                        # the standard renderer below instead of emitting an
                        # empty box.

                    if _is_multilevel and isinstance(p.solutions, dict) and all(isinstance(v, dict) for v in p.solutions.values()):
                        _dirs = getattr(p, 'directions', None)
                        for level_name, level_vars in p.solutions.items():
                            _idx = list(p.solutions.keys()).index(level_name)
                            _val = p.objective_values[0][_idx] if _idx < p.objective_values.shape[1] else None
                            _colored = _colorize_obj_values([_val], [_dirs[_idx]] if _dirs and _idx < len(_dirs) else None) if _val is not None else [format_string(_val)]
                            box.row(left=f"Level: {level_name}:",
                                    right=' '.join(_colored))
                            for key in level_vars:
                                if key is None or not isinstance(key, str):
                                    continue
                                if not hidden_variables and (key.startswith('_') or 'autolin' in key or 'sos2' in key or 'rdiv' in key):
                                    continue
                                if not key_vars or key in key_vars:
                                    if show_elements:
                                        box.print_element(key, level_vars[key])
                                    else:
                                        box.print_packed_tensors([(key, level_vars[key])])
                    elif p.number_of_objectives == 1:
                        try:
                            _dirs = getattr(p, 'directions', None)
                            _d = _dirs[0] if _dirs and len(_dirs) > 0 else ""
                            _val = format_string(p.objective_values[0][0])
                            _obj_right = f"{_d}: {_val}" if _d else _val
                            box.row(left='Objective', right=_obj_right)
                        except Exception:
                            pass
                        # named terms (m.term) always render in the Decision
                        # box — key_vars only restricts plain variables
                        _term_names = set()
                        try:
                            _tf = getattr(p.em, 'features', None) or {}
                            _term_names = set(_tf.get('_term_order') or [])
                            _term_names |= set(_tf.get('_terms') or {})
                            _term_names |= set(_tf.get('_deferred_terms') or {})
                        except Exception:
                            pass
                        _packed_vars = []
                        for key in p.solutions:
                            if key is None or not isinstance(key, str):
                                continue
                            if not hidden_variables and (key.startswith('_') or 'autolin' in key or 'sos2' in key or 'rdiv' in key):
                                continue
                            if not key_vars or key in key_vars or key in _term_names:
                                if show_elements:
                                    box.print_element(key, p.solutions[key])
                                else:
                                    _packed_vars.append((key, p.solutions[key]))
                        if _packed_vars and not show_elements:
                            box.print_packed_tensors(_packed_vars)
                    else:
                        _dirs = getattr(p, 'directions', None)
                        for i in range(p.objective_values.shape[0]):
                            if not skip or i % skip == 0:
                                _obj_parts = []
                                for j in range(p.objective_values.shape[1]):
                                    _d = _dirs[j] if _dirs and j < len(_dirs) else ""
                                    _val = format_string(p.objective_values[i][j])
                                    _obj_parts.append(f"{_d}: {_val}" if _d else _val)
                                box.row(
                                    left=f"Pareto {i}",
                                    right=', '.join(_obj_parts),
                                )
                                _packed_vars = []
                                for key in p.solutions[i]:
                                    if key is None or not isinstance(key, str):
                                        continue
                                    if not hidden_variables and (key.startswith('_') or 'autolin' in key or 'sos2' in key or 'rdiv' in key):
                                        continue
                                    if not key_vars or key in key_vars:
                                        if show_elements:
                                            box.print_element(key, p.solutions[i][key])
                                        else:
                                            _packed_vars.append((key, p.solutions[i][key]))
                                if _packed_vars and not show_elements:
                                    box.print_packed_tensors(_packed_vars)

                    box.bottom()
                except:
                    box.bottom()
        else:
            p.em.show_tensor = not show_elements
            p.em.show_detailed_tensors = False
            p.em.output_decimals = 4
            ReportEngine(p.em)._generate_decision_info(box=box)

        if not box._buffered:
            box.render()

    def report_binding_constraints(self, style=1, width=78, box=None):
        p = self._p
        if p.method == 'madm':
            return
        if not p.healthy():
            return
        if p.method == "heuristic":
            return
        if not p.em.features.get('constraint_labels'):
            return
        if p.number_of_objectives != 1:
            return

        binding = []
        non_binding = []
        try:
            for label in p.em.features['constraint_labels']:
                if label.startswith('_'):
                    continue
                try:
                    slack_val = p.em.get_slack(label)
                    if slack_val is not None and abs(slack_val) < 1e-6:
                        binding.append(label)
                    else:
                        non_binding.append(label)
                except:
                    non_binding.append(label)
        except:
            return

        if not binding:
            return

        standalone = box is None
        if standalone:
            box = report(width=width, style=style)

        box.top(left="Binding Constraints")
        _stderr_fd = sys.stderr.fileno()
        _saved_stderr = os.dup(_stderr_fd)
        _devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull, _stderr_fd)
        os.close(_devnull)
        try:
            for label in binding:
                try:
                    dual_val = p.em.get_dual(label, tensor=False)
                    dual_str = format_string(dual_val) if dual_val is not None else "n/a"
                    box.row(left=f"{label}:", right=f"slack=0  dual={dual_str}")
                except:
                    box.row(left=f"{label}:", right="slack=0")
        finally:
            os.dup2(_saved_stderr, _stderr_fd)
            os.close(_saved_stderr)
        box.bottom()

        if standalone:
            box.render()

    def report_metrics(self, style=1, width=78, skip=False, box=None):
        p = self._p
        if box is None:
            box = report(width=width, style=style)

        _is_sensitivity = getattr(p, 'sensitivity_analyzed', False)

        stg_str = "-"
        try:
            if p.track_history and p.method == "heuristic":
                stg_val = getattr(p, 'stagnation', None)
                if stg_val is not None:
                    stg_str = format_string(stg_val, ensure_length=True)
        except Exception:
            pass

        ogr_str = "-"
        try:
            if _is_sensitivity:
                ogr_vals = []
                for pname in getattr(p, 'sensitivity_parameter_names', []):
                    key = f"sensitivity_of_ogr_to_{pname}"
                    if key in getattr(p, 'sensitivity_data', {}):
                        ogr_vals.extend([v for v in p.sensitivity_data[key] if v is not None])
                if ogr_vals:
                    ogr_str = format_string(sum(ogr_vals) / len(ogr_vals))
            else:
                ogr_val = p.get_ogr()
                if ogr_val is not None and ogr_val != 0:
                    ogr_str = format_string(ogr_val)
        except Exception:
            pass

        nzr_str = "-"
        try:
            total_var_count = (
                p.em.features.get("total_variable_counter", [0, 0])[1]
                if hasattr(p.em, "features") and isinstance(p.em.features, dict)
                else 0
            )
            total_density = p.get_density()
            if total_density != "n/a" and total_var_count != 0:
                nzr_str = format_string(total_density / total_var_count)
        except Exception:
            pass

        pvr_str = "-"
        try:
            if p.dataset_size and p.dataset_size > 0:
                total_var = p.em.features.get('total_variable_counter', [0, 0])[1]
                if total_var > 0:
                    pvr_str = format_string(p.dataset_size / total_var)
        except Exception:
            pass

        cfm_str = "-"
        try:
            if p.track_history and p.method == "heuristic":
                cfm_val = getattr(p, 'convergence_rate', None)
                if cfm_val is not None:
                    cfm_str = format_string(cfm_val)
            elif p.number_of_objectives > 1:
                cfm = getattr(p, 'conflict_metric', None)
                if cfm is not None:
                    cfm_str = format_string(cfm)
        except Exception:
            pass

        imp_str = "-"
        try:
            if p.track_history and p.method == "heuristic":
                imp_val = getattr(p, 'improvement_pct', None)
                if imp_val is not None:
                    imp_str = format_string(imp_val)
        except Exception:
            pass

        div_str = "-"
        try:
            if p.track_history and p.method == "heuristic":
                div_val = getattr(p, 'population_diversity', None)
                if div_val is not None:
                    div_str = format_string(div_val)
        except Exception:
            pass

        try:
            if _is_sensitivity:
                cpt_vals = []
                for pname in getattr(p, 'sensitivity_parameter_names', []):
                    key = f"sensitivity_of_cpt_to_{pname}"
                    if key in getattr(p, 'sensitivity_data', {}):
                        cpt_vals.extend([v for v in p.sensitivity_data[key] if v is not None])
                avg_cpt = sum(cpt_vals) / len(cpt_vals) if cpt_vals else p.cpt
                cpt_str = format_time_duration(avg_cpt)
                cpt_raw = avg_cpt
                mgt_str = format_time_duration(max(p.mgt, 0))
                mgt_raw = max(p.mgt, 0)
            else:
                mgt_str = format_time_duration(p.mgt)
                mgt_raw = p.mgt
                cpt_str = format_time_duration(p.cpt)
                cpt_raw = p.cpt
        except Exception:
            mgt_str = "-"
            mgt_raw = "-"
            cpt_str = "-"
            cpt_raw = "-"

        _problem_size = 0
        try:
            if hasattr(p.em, 'features') and isinstance(p.em.features, dict):
                _var_count = p.em.features.get('total_variable_counter', [0, 0])[1]
                _con_count = p.em.features.get('constraint_counter', [0, 0])[1]
                _obj_count = p.em.features.get('objective_counter', [0, 0])[1]
                _problem_size = _var_count + _con_count + _obj_count
        except Exception:
            pass

        all_headers_full = ["STG", "CVR", "IMP", "DIV", "OGR", "PVR", "NZR", "MGT", "CPT"]
        all_values_full = [stg_str, cfm_str, imp_str, div_str, ogr_str, pvr_str, nzr_str, mgt_str, cpt_str]
        all_raws_full = [stg_str, cfm_str, imp_str, div_str, ogr_str, pvr_str, nzr_str, mgt_raw, cpt_raw]
        all_expl_full = [
            "Stagnation",
            "Convergence Rate",
            "Improvement %",
            "Diversity",
            "Gap Ratio",
            "Pareto Ratio",
            "Sparsity",
            "Gen. Time",
            "Comp. Time",
        ]

        visible = [(h, v, r, e) for h, v, r, e in zip(all_headers_full, all_values_full, all_raws_full, all_expl_full) if v != "-"]
        if not visible:
            visible = [(all_headers_full[5], all_values_full[5], all_raws_full[5], all_expl_full[5]),
                       (all_headers_full[6], all_values_full[6], all_raws_full[6], all_expl_full[6])]

        all_labels = []
        for h, v in zip([x[0] for x in visible], [x[2] for x in visible]):
            if h in ("MGT", "CPT"):
                all_labels.append(_metric_label(h, v, problem_size=_problem_size))
            else:
                all_labels.append(_metric_label(h, v))

        # inner_width = content area between the two border chars (excludes
        # the single padding space on each side that every row adds).
        # A row is emitted as: border + " " + inner + " " + border
        # so inner must be exactly (width - 4) chars wide.
        inner_width = width - 4          # e.g. 74 for width=78
        half = (len(visible) + 1) // 2
        sep = " │ "
        sep_w = len(sep)                 # 3
        # left_col gets the extra char when inner_width - sep_w is odd
        left_col_w = (inner_width - sep_w + 1) // 2   # ceil, e.g. 36
        right_col_w = inner_width - sep_w - left_col_w # floor, e.g. 35

        # In data rows: "│ " + left_cell(36) + " │ " + right_cell(35) + " │"
        # The middle │ appears at absolute position from line start:
        # 1 + 1 + 36 + 1 + 1 = 40 (for width=78)
        # In border lines: corner(1) + bars + ┬/┴ + bars + corner(1) = width
        # The ┬/┴ must appear at position 40, so corner(1) + bars_before = 39
        mid_sep_position = 1 + 1 + left_col_w + 1 + 1  # absolute column of middle │
        bars_before_sep = mid_sep_position - 2  # subtract corner and ┬/┴ itself

        has_right = len(visible) > 1

        if has_right:
            label = " Performance "
            # Top line: corner + " " + label + bars + ┬ + bars + corner = width
            # We want the label on the left side, then bars until the ┬
            # The ┬ must appear at position mid_sep_position
            # Structure: corner(1) + " "(1) + label(len) + bars + ┬(1) at position mid_sep_position
            # So: 1 + 1 + len(label) + bars_to_sep = mid_sep_position - 1
            # bars_to_sep = mid_sep_position - 1 - 1 - 1 - len(label) = mid_sep_position - 3 - len(label)
            bars_to_sep = mid_sep_position - 3 - len(label)
            if bars_to_sep < 0:
                bars_to_sep = 0
            # bars_after: from ┬ to end
            bars_after = width - 2 - 1 - len(label) - bars_to_sep - 1
            if bars_after < 0:
                bars_after = 0
            top_line = (
                f"{box._topleft}"
                f"{box._topbar}"
                f"{label}"
                f"{box._topbar * bars_to_sep}"
                f"{box._middletop}"
                f"{box._topbar * bars_after}"
                f"{box._topright}"
            )
            box._emit(top_line, "top")
        else:
            box.top(left="Performance")

        for i in range(half):
            lh, lv, lr, le = visible[i]
            lbl_l = all_labels[i]

            left_name = lh[:4].ljust(4)
            left_val = lv.strip().rjust(14)
            left_status = lbl_l.rjust(left_col_w - 20)

            left_cell = (left_name + " " + left_val + " " + left_status).ljust(left_col_w)

            if i + half < len(visible):
                rh, rv, rr, re = visible[i + half]
                lbl_r = all_labels[i + half]
                right_name = rh[:4].ljust(4)
                right_val = rv.strip().rjust(14)
                right_status = lbl_r.rjust(right_col_w - 20)
                right_cell = (right_name + " " + right_val + " " + right_status).ljust(right_col_w)
                row_sep = sep
            else:
                right_cell = " " * right_col_w
                if half <= 1:
                    row_sep = sep[0] + " " + sep[2]  # hide │ for single-row box
                else:
                    row_sep = sep

            inner = left_cell + row_sep + right_cell
            # safety pad (should be zero if math is right)
            pad = inner_width - len(inner)
            if pad > 0:
                inner += " " * pad
            box._emit(f"{box._border} {inner} {box._border}", "row")

        if has_right:
            # Bottom line: corner + bars + ┴ + bars + corner = width
            # The ┴ must appear at position mid_sep_position (same as rows)
            # corner(1) + bars_before + ┴ at position mid_sep_position
            # So: 1 + bars_before = mid_sep_position - 1
            bars_before = mid_sep_position - 2  # position before ┴
            bars_after = width - 2 - bars_before - 1  # total - corners - bars_before - ┴
            bot_line = (
                f"{box._bottomleft}"
                f"{box._bottombar * bars_before}"
                f"{box._middlebottom}"
                f"{box._bottombar * bars_after}"
                f"{box._bottomright}"
            )
            box._emit(bot_line, "bottom")
        else:
            box.bottom()

        if not box._buffered:
            box.render()

    def report_decomposition(self, style=1, width=78, box=None):
        """Report decomposition-specific metrics (Benders, Lagrangian, CG, Branching)."""
        p = self._p
        if not getattr(p, 'boost', False):
            return

        benders_res = getattr(p, '_benders_result', None)
        lagrangian_res = getattr(p, '_lagrangian_result', None)
        cg_res = getattr(p, '_cg_result', None)
        branching_res = getattr(p, '_branching_result', None)

        has_decomp = any(r is not None for r in [
            benders_res, lagrangian_res, cg_res, branching_res])
        if not has_decomp:
            return

        standalone = box is None
        if standalone:
            box = report(width=width, style=style)

        from ..algorithms.exact.benders.result import BendersResult as _BR
        from ..algorithms.exact.benders.enums import BendersStatus as _BS

        if isinstance(benders_res, _BR):
            status_val = benders_res.status
            if hasattr(status_val, 'value'):
                status_val = status_val.value
            box.top(left="Decomposition")
            box.row(left="Method", right="Benders")
            box.row(left="Status", right=str(status_val))
            box.row(left="Iterations", right=str(benders_res.iterations))
            box.row(left="Cuts",
                    right=f"{benders_res.n_optimality_cuts} opt  "
                          f"{benders_res.n_feasibility_cuts} feas  "
                          f"{benders_res.n_cuts} total")
            box.row(left="Solutions", right=str(benders_res.n_sol))
            lb = benders_res.lb_history[-1] if benders_res.lb_history else None
            ub = benders_res.ub_history[-1] if benders_res.ub_history else None
            if lb is not None and ub is not None:
                box.row(left="Bounds",
                        right=f"LB={lb:.4g}  UB={ub:.4g}")
            if benders_res.gap_rel < float('inf'):
                box.row(left="Gap",
                        right=f"{benders_res.gap_rel:.2e}")
            rt = benders_res.runtime_total
            rm = benders_res.runtime_master
            rs = benders_res.runtime_subproblem
            if rt > 0:
                box.row(left="Runtime",
                        right=f"total={rt:.2f}s  "
                              f"master={rm:.2f}s  sub={rs:.2f}s")
            box.bottom()

        elif lagrangian_res is not None:
            status_val = getattr(lagrangian_res, 'status', 'unknown')
            if hasattr(status_val, 'value'):
                status_val = status_val.value
            box.top(left="Decomposition")
            box.row(left="Method", right="Lagrangian Relaxation")
            box.row(left="Status", right=str(status_val))
            iters = getattr(lagrangian_res, 'iterations', 0)
            box.row(left="Iterations", right=str(iters))
            lb = getattr(lagrangian_res, 'lower_bound', None)
            ub = getattr(lagrangian_res, 'upper_bound', None)
            if lb is not None and ub is not None:
                box.row(left="Bounds",
                        right=f"LB={lb:.4g}  UB={ub:.4g}")
            rt = getattr(lagrangian_res, 'runtime_total', 0)
            if rt > 0:
                box.row(left="Runtime", right=f"{rt:.2f}s")
            box.bottom()

        elif cg_res is not None:
            status_val = getattr(cg_res, 'status', 'unknown')
            if hasattr(status_val, 'value'):
                status_val = status_val.value
            box.top(left="Decomposition")
            box.row(left="Method", right="Column Generation")
            box.row(left="Status", right=str(status_val))
            iters = getattr(cg_res, 'iterations', 0)
            box.row(left="Iterations", right=str(iters))
            n_cols = getattr(cg_res, 'n_columns_added', 0)
            if n_cols > 0:
                box.row(left="Columns Added", right=str(n_cols))
            lb = getattr(cg_res, 'lower_bound', None)
            ub = getattr(cg_res, 'upper_bound', None)
            if lb is not None and ub is not None and lb > float('-inf'):
                box.row(left="Bounds",
                        right=f"LB={lb:.4g}  UB={ub:.4g}")
            rt = getattr(cg_res, 'runtime_total', 0)
            try:
                rt = float(rt)
            except (TypeError, ValueError):
                rt = 0
            if rt > 0:
                box.row(left="Runtime", right=f"{rt:.2f}s")
            box.bottom()

        elif branching_res is not None:
            status_val = getattr(branching_res, 'status', 'unknown')
            if hasattr(status_val, 'value'):
                status_val = status_val.value
            box.top(left="Decomposition")
            box.row(left="Method", right="Branch and Bound")
            box.row(left="Status", right=str(status_val))
            iters = getattr(branching_res, 'iterations', 0)
            n_nodes = getattr(branching_res, 'n_nodes_explored', 0)
            if n_nodes > 0:
                box.row(left="Nodes Explored", right=str(n_nodes))
            elif iters > 0:
                box.row(left="Iterations", right=str(iters))
            n_cuts = getattr(branching_res, 'n_cuts_added',
                              getattr(branching_res, 'n_cuts', 0))
            if n_cuts > 0:
                box.row(left="Cuts Added", right=str(n_cuts))
            n_cols = getattr(branching_res, 'n_columns_generated', 0)
            if n_cols > 0:
                box.row(left="Columns Generated", right=str(n_cols))
            gap_val = getattr(branching_res, 'gap', None)
            if gap_val is not None and gap_val < float('inf'):
                box.row(left="Gap", right=f"{gap_val:.2e}")
            n_sol = getattr(branching_res, 'solution_count', 0)
            if n_sol > 0:
                box.row(left="Solutions Found", right=str(n_sol))
            rt = getattr(branching_res, 'runtime_total', 0)
            if rt > 0:
                box.row(left="Runtime", right=f"{rt:.2f}s")
            box.bottom()

        if standalone:
            box.render()

    def report_data(self, style=1, width=78, box=None):
        p = self._p
        if box is None:
            box = report(width=width, style=style)
        if p.inputdata:
            if type(p.inputdata)!=dict:
                box.top(left="Data")
                try:
                    col_headers = ["Size", "Min", "Max", "Ave", "Std"]
                    all_rows_data = []
                    for name in p.inputdata.data.keys():
                        row_vals = [
                            p.inputdata.type_params[name].strip(),
                            format_string(p.inputdata.size_params[name]),
                            format_string(p.inputdata.minimum_params[name]),
                            format_string(p.inputdata.maximum_params[name]),
                            format_string(p.inputdata.average_params[name]),
                            format_string(p.inputdata.std_params[name]),
                        ]
                        all_rows_data.append((name, row_vals))

                    all_strings = [col_headers] + [vals[1:] for _, vals in all_rows_data]
                    col_widths = [max(len(row[i]) for row in all_strings) for i in range(len(col_headers))]

                    row_cells_header = "  ".join(h.rjust(col_widths[i]) for i, h in enumerate(col_headers))

                    label_part = "Domain" + "  "
                    pad_header = width - 4 - len(label_part) - len(row_cells_header)
                    if pad_header < 0:
                        pad_header = 0
                    inner_header = label_part + " " * pad_header + row_cells_header
                    box._emit(f"{box._border} {inner_header} {box._border}", "row")

                    for name, vals in all_rows_data:
                        domain_val = vals[0]
                        numeric_vals = vals[1:]
                        row_cells_vals = "  ".join(v.rjust(col_widths[i]) for i, v in enumerate(numeric_vals))
                        name_display = f"{name}"
                        label_part = name_display + "  " + domain_val + "  "
                        pad_vals = width - 4 - len(label_part) - len(row_cells_vals)
                        if pad_vals < 0:
                            pad_vals = 0
                        inner_vals = label_part + " " * pad_vals + row_cells_vals
                        box._emit(f"{box._border} {inner_vals} {box._border}", "row")
                except Exception:
                    pass
                box.bottom()

        if not box._buffered:
            box.render()

    def report_benchmark(self, width=78, style=1, box=None):
        p = self._p
        if box is None:
            box = report(width=width, style=style)
        if p.should_benchmark and hasattr(p, 'ben_results') and p.ben_results is not None and len(p.ben_results) > 0:
            df = p.ben_results.clone()

            # Sort alphabetically by interface then solver
            try:
                df = df.sort(['interface', 'solver'])
            except Exception:
                pass

            box.top(left="Benchmark")

            inner_w = width - 4  # content between border chars

            # Columns: Rank | Interface | Solver | Time Ave | Time Std | Obj Ave | Obj Std
            # Fixed widths that sum to inner_w
            _c = [4, 12, 12, 10, 10, 12, 12]   # sum = 72 = inner_w for width=78 (74 - 2 separators)
            # Recompute dynamically to fill exactly inner_w
            _sep = " │"
            _sep_w = len(_sep)  # 2
            _n_sep = len(_c) - 1
            _total_sep = _n_sep * _sep_w
            _data_w = inner_w - _total_sep
            # Distribute: rank=4, iface=proportional, solver=proportional, rest=fixed
            _rank_w  = 4
            _time_w  = 9
            _obj_w   = 11
            _remain  = _data_w - _rank_w - 2 * _time_w - 2 * _obj_w
            _iface_w = _remain // 2
            _solver_w = _remain - _iface_w
            _widths  = [_rank_w, _iface_w, _solver_w, _time_w, _time_w, _obj_w, _obj_w]

            def _cell(txt, w, align='left'):
                txt = str(txt)
                if len(txt) > w:
                    txt = txt[:w-1] + '…'
                return txt.ljust(w) if align == 'left' else txt.rjust(w)

            def _row(*cells):
                parts = [_cell(cells[i], _widths[i], 'right' if i >= 3 else 'left') for i in range(len(_widths))]
                inner = (" │".join(parts))
                pad = inner_w - len(inner)
                if pad > 0:
                    inner += " " * pad
                box._emit(f"{box._border} {inner} {box._border}", "row")

            # Header
            _row("#", "Interface", "Solver", "T.Ave", "T.Std", "Obj.Ave", "Obj.Std")

            # Separator
            _sep_line_parts = [box._topbar * w for w in _widths]
            _sep_inner = ("─┼".join(_sep_line_parts))
            pad = inner_w - len(_sep_inner)
            if pad > 0:
                _sep_inner += box._topbar * pad
            box._emit(f"{box._border} {_sep_inner} {box._border}", "row")

            # Data rows
            for idx, row in enumerate(df.iter_rows(named=True)):
                try:
                    iface   = str(row.get('interface', '-'))
                    solver  = str(row.get('solver',    '-'))
                    t_ave   = row.get('time_ave', None) if 'time_ave' in df.columns else row.get('time', None)
                    t_std   = row.get('time_std', None) if 'time_std' in df.columns else None
                    o_ave   = row.get('obj_ave', None) if 'obj_ave' in df.columns else row.get('obj', None)
                    o_std   = row.get('obj_std', None) if 'obj_std' in df.columns else None

                    def _fmt_t(v):
                        try: return format_time_duration(float(v), name="")
                        except Exception: return "-"
                    def _fmt_n(v, dp=2):
                        try:
                            fv = float(v)
                            if abs(fv) >= 1e6 or (abs(fv) < 1e-3 and fv != 0):
                                return f"{fv:.2e}"
                            return f"{fv:.{dp}f}"
                        except Exception: return "-"

                    _row(
                        str(idx + 1),
                        iface, solver,
                        _fmt_t(t_ave), _fmt_t(t_std),
                        _fmt_n(o_ave), _fmt_n(o_std),
                    )
                except Exception:
                    pass

            box.bottom()

        if not box._buffered:
            box.render()

    def report_info(self, width=78, style=1, box=None):
        p = self._p
        if box is None:
            box = report(width=width, style=style)

        _has_sens     = getattr(p, 'sensitivity_analyzed', False)
        _has_bench    = getattr(p, 'should_benchmark', False) and hasattr(p, 'ben_results')
        _method       = getattr(p, 'method', 'exact')
        _n_obj        = getattr(p, 'number_of_objectives', 1)
        _is_heuristic = _method in ('heuristic', 'pso', 'de', 'ga')
        _is_multiobj  = _n_obj > 1
        _has_int      = _has_sens and bool(getattr(p, 'sensitivity_interaction_data', []))

        inner_w = width - 4
        # Layout: abbrev(5) │ range(10) │ dir(1) │ description fills rest
        _a = 5    # abbrev
        _r = 10   # range
        _d = 1    # direction arrow
        _sep = " │ "
        _desc_w = inner_w - _a - _r - _d - 3 * len(_sep)

        def _sec(title):
            bar = "─" * (inner_w - len(title) - 1)
            box._emit(f"{box._border} {title} {bar} {box._border}", "row")

        def _blank():
            box._emit(f"{box._border} {' ' * inner_w} {box._border}", "row")

        def _entry(abbrev, rng, direction, description):
            a = abbrev.ljust(_a)
            r = rng.ljust(_r)
            d = direction          # single char: ↓ ↑ –
            desc = description[:_desc_w].ljust(_desc_w)
            inner = f"{a}{_sep}{r}{_sep}{d}{_sep}{desc}"
            pad = inner_w - len(inner)
            if pad > 0:
                inner += " " * pad
            box._emit(f"{box._border} {inner} {box._border}", "row")

        def _col_header():
            a = "Abbr".ljust(_a)
            r = "Range".ljust(_r)
            d = "↕"
            desc = "Description".ljust(_desc_w)
            inner = f"{a}{_sep}{r}{_sep}{d}{_sep}{desc}"
            pad = inner_w - len(inner)
            if pad > 0:
                inner += " " * pad
            box._emit(f"{box._border} {inner} {box._border}", "row")

        box.top(left="Information")
        _col_header()

        # ── Data ────────────────────────────────────────────────────────
        _blank()
        _sec("Data")
        _entry("T",   "–",       "–", "Data type: S=scalar, V=vector, M=matrix")
        _entry("S",   "ℕ",       "–", "Total number of elements in the dataset")
        _entry("NZ",  "ℕ",       "–", "Count of non-zero entries")
        _entry("Min", "ℝ",       "–", "Smallest value in the dataset")
        _entry("Max", "ℝ",       "–", "Largest value in the dataset")
        _entry("Ave", "ℝ",       "–", "Arithmetic mean of the dataset")
        _entry("Std", "[0, ∞)",  "–", "Spread around the mean (standard deviation)")

        # ── Performance ─────────────────────────────────────────────────
        _blank()
        _sec("Performance")
        _entry("CPT", "[0, ∞)",  "↓", "Solver wall-clock time; lower is faster")
        if _is_heuristic:
            _entry("CVR", "[0, 1]",  "↑", "Fraction of epochs with objective improvement")
            _entry("DIV", "[0, ∞)",  "↑", "Average spread of population; higher = more explored")
            _entry("IMP", "[0, ∞)",  "↑", "Mean objective gain per epoch; higher = faster learning")
        _entry("MGT", "[0, ∞)",  "↓", "Time to build the model before solving")
        _entry("NZR", "[0, 1]",  "↓", "Fraction of non-zero constraint coefficients")
        _entry("OGR", "[0, 1]",  "↓", "Gap between best bound and best solution; 0 = optimal")
        _entry("PVR", "[0, ∞)",  "↓", "Dataset size divided by variable count; data-richness")
        if _is_heuristic:
            _entry("STG", "[0, 1]",  "↓", "Fraction of epochs without improvement; 0 = active search")
        if _is_multiobj:
            _entry("CFM", "[0, 1]",  "↓", "Degree of conflict between objectives; 0 = aligned")

        # ── Sensitivity ─────────────────────────────────────────────────
        if _has_sens:
            _blank()
            _sec("Sensitivity")
            _entry("Asy", "[0, 1]",  "↓", "Asymmetry: how differently the objective responds up vs down")
            _entry("CPT", "Δ%",      "–", "% change in solve time relative to base scenario")
            _entry("Dec", "[0, 1]",  "↑", "Fraction of decision variables that shifted value")
            _entry("Imp", "[0, 1]",  "↑", "Combined score: Sen × Dec; overall parameter importance")
            _entry("Obj", "Δ%",      "–", "% change in objective value relative to base scenario")
            _entry("Sen", "[0, 1]",  "↑", "Elasticity: % obj change per % parameter change")
            _entry("Tim", "[0, 1]",  "↓", "Variability of solve time across parameter scenarios")
            _entry("Unc", "[0, 1]",  "↓", "Coefficient of variation of objective across scenarios")
            _entry("Val", "Δ%",      "–", "% change in parameter value relative to base scenario")

            if _has_int:
                _blank()
                _sec("Interaction")
                _entry("Dec", "[0, 1]",  "↑", "Fraction of decisions changed when both params shift jointly")
                _entry("Non", "[0, 1]",  "↑", "Non-additivity: how much joint effect exceeds sum of singles")
                _entry("Str", "[0, 1]",  "↑", "Joint interaction effect relative to individual main effects")

        if _has_bench:
            _blank()
            _sec("Benchmark")
            _entry("T.Ave", "[0, ∞)",  "↓", "Mean solve time across repeated runs")
            _entry("T.Std", "[0, ∞)",  "↓", "Std dev of solve time; lower = more consistent")
            _entry("O.Ave", "ℝ",       "–", "Mean objective value across repeated runs")
            _entry("O.Std", "[0, ∞)",  "↓", "Std dev of objective; lower = more reproducible")

        box.bottom()

        if not box._buffered:
            box.render()

    def report_sensitivity(self, width=78, style=1, skip=False, show_elements=False, box=None, hidden_variables=False):

        p = self._p
        if box is None:
            box = report(width=width, style=style)
        if not p.sensitivity_analyzed:
            if not box._buffered:
                box.render()
            return

        import numpy as _np
        sd = p.sensitivity_data
        _descriptions = getattr(p, 'sensitivity_descriptions', {})
        _is_single = p.number_of_objectives == 1
        _directions = getattr(p, 'directions', []) or []
        _obj_dir = _directions[0] if _directions else 'min'
        _var_types = getattr(p, 'em', None)
        _var_types = _var_types.features.get('variable_type', {}) if _var_types else {}
        _obj_weights = getattr(p, 'sensitivity_obj_weights', None)

        def _aggregate_objective(obj_val):
            if obj_val is None:
                return None
            if _is_single:
                try:
                    return float(obj_val)
                except (TypeError, ValueError):
                    return None
            arr = _np.asarray(obj_val, dtype=float)
            if arr.ndim == 0:
                return float(arr)
            if arr.size == 0:
                return None
            if _obj_weights is not None:
                w = _np.asarray(_obj_weights, dtype=float)
                if w.shape == arr.shape:
                    return float(_np.sum(arr * w) / _np.sum(w))
            return float(_np.mean(arr))

        def _is_array_val(v):
            return isinstance(v, (_np.ndarray, list, tuple))

        def _format_base_val(v):
            if _is_array_val(v):
                arr = _np.asarray(v, dtype=float)
                _mean = _np.mean(arr)
                _std = _np.std(arr)
                return f"{format_string(_mean)} \u00b1 {format_string(_std)}"
            return format_string(v)

        def _summarize_var(v, vtype=None):
            if v is None:
                return "-"
            if isinstance(v, _np.ndarray):
                if v.size == 1:
                    return format_string(float(v.flat[0]))
                if vtype == 'pvar':
                    _mean = _np.mean(v)
                    _std = _np.std(v)
                    return f"{format_string(_mean)} \u00b1 {format_string(_std)}"
                if vtype == 'ivar':
                    _total = _np.sum(v)
                    return f"{format_string(_total)}/{v.size}"
                if vtype == 'bvar':
                    if v.ndim == 2:
                        _row_nz = _np.count_nonzero(v, axis=1)
                        _col_nz = _np.count_nonzero(v, axis=0)
                        return f"r{_row_nz.tolist()}c{_col_nz.tolist()}"
                    elif v.ndim >= 3:
                        _total = int(_np.sum(v))
                        return f"{_total}/{v.size}"
                _nz = int(_np.count_nonzero(v))
                return f"{_nz}/{v.size}"
            if isinstance(v, (int, float)):
                return format_string(v)
            if isinstance(v, (list, tuple)):
                arr = _np.asarray(v, dtype=float)
                if vtype == 'pvar':
                    _mean = _np.mean(arr)
                    _std = _np.std(arr)
                    return f"{format_string(_mean)} \u00b1 {format_string(_std)}"
                if vtype == 'ivar':
                    _total = _np.sum(arr)
                    return f"{format_string(_total)}/{len(v)}"
                if vtype == 'bvar':
                    if arr.ndim == 2:
                        _row_nz = _np.count_nonzero(arr, axis=1)
                        _col_nz = _np.count_nonzero(arr, axis=0)
                        return f"r{_row_nz.tolist()}c{_col_nz.tolist()}"
                    elif arr.ndim >= 3:
                        _total = int(_np.sum(arr))
                        return f"{_total}/{arr.size}"
                _nz = int(_np.count_nonzero(arr))
                return f"{_nz}/{len(v)}"
            return format_string(v)

        def _pct_str(val, base):
            if val is None or base is None:
                return "-"
            try:
                base_f = float(base)
                val_f = float(val)
                if base_f == 0:
                    v = val_f
                else:
                    v = (val_f - base_f) / abs(base_f) * 100
                for dec in (2, 1, 0):
                    s = f"{v:+.{dec}f}"
                    if len(s) <= 6:
                        return s
                for dec in (1, 0):
                    s = f"{v:+.{dec}e}"
                    if len(s) <= 6:
                        return s
                return f"{v:+.0e}"
            except (TypeError, ValueError, ZeroDivisionError):
                return "-"

        def _pval_pct_str(val, base):
            if val is None or base is None:
                return "-"
            try:
                if _is_array_val(val) and _is_array_val(base):
                    val_arr = _np.asarray(val, dtype=float)
                    base_arr = _np.asarray(base, dtype=float)
                    
                    is_coords_val = val_arr.ndim == 2 and val_arr.shape[1] == 2
                    is_coords_base = base_arr.ndim == 2 and base_arr.shape[1] == 2
                    
                    if is_coords_val and is_coords_base:
                        val_n = val_arr.shape[0]
                        base_n = base_arr.shape[0]
                        
                        centroid = _np.mean(base_arr, axis=0)
                        val_spread = _np.mean(_np.abs(val_arr - centroid))
                        base_spread = _np.mean(_np.abs(base_arr - centroid))
                        if base_spread < 1e-10:
                            return "+0%"
                        ratio = val_spread / base_spread
                        pct = (ratio - 1.0) * 100
                        return f"{pct:+.0f}%"
                    elif is_coords_val or is_coords_base:
                        if is_coords_val:
                            return f"n={val_arr.shape[0]}"
                        else:
                            return f"n={base_arr.shape[0]}"
                    
                    v_mean = float(_np.mean(val_arr))
                    b_mean = float(_np.mean(base_arr))
                elif _is_array_val(val):
                    v_mean = float(_np.mean(_np.asarray(val, dtype=float)))
                    b_mean = float(base)
                elif _is_array_val(base):
                    v_mean = float(val)
                    b_mean = float(_np.mean(_np.asarray(base, dtype=float)))
                else:
                    v_mean = float(val)
                    b_mean = float(base)
                def _fmt_pct(v):
                    for dec in (2, 1, 0):
                        s = f"{v:+.{dec}f}"
                        if len(s) <= 6:
                            return s
                    for dec in (1, 0):
                        s = f"{v:+.{dec}e}"
                        if len(s) <= 6:
                            return s
                    return f"{v:+.0e}"

                if b_mean == 0:
                    s = _fmt_pct(v_mean)
                else:
                    pct = (v_mean - b_mean) / abs(b_mean) * 100
                    s = _fmt_pct(pct)
                return s
            except (TypeError, ValueError, ZeroDivisionError):
                return "-"

        def _count_nz(sol):
            if not isinstance(sol, dict):
                return 0
            ct = 0
            for k, v in sol.items():
                if k.startswith('_') or 'autolin' in k or 'sos2' in k or 'rdiv' in k:
                    continue
                if isinstance(v, _np.ndarray):
                    ct += int(_np.count_nonzero(v))
                elif isinstance(v, (int, float)):
                    ct += 1 if v != 0 else 0
                elif isinstance(v, (list, tuple)):
                    ct += sum(1 for x in v if x != 0)
            return ct

        def _sol_similarity(sol_a, sol_b):
            if not isinstance(sol_a, dict) or not isinstance(sol_b, dict):
                return None
            total = 0
            same = 0
            for k in sol_a:
                if k.startswith('_') or 'autolin' in k or 'sos2' in k or 'rdiv' in k:
                    continue
                if k not in sol_b:
                    continue
                va, vb = sol_a[k], sol_b[k]
                try:
                    if isinstance(va, _np.ndarray) and isinstance(vb, _np.ndarray):
                        total += va.size
                        same += int(_np.sum(va == vb))
                    elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                        total += 1
                        same += 1 if va == vb else 0
                    elif isinstance(va, (list, tuple)) and isinstance(vb, (list, tuple)):
                        total += len(va)
                        same += sum(1 for a, b in zip(va, vb) if a == b)
                except Exception:
                    pass
            if total == 0:
                return None
            return same / total * 100

        def _control_idx_for(pname):
            # where the control (baseline) scenario sits in the recorded
            # scenario list — set by report_sensitivity; the constructor
            # default (0) is wrong whenever scenarios are generated sorted
            _ci = getattr(p, 'sensitivity_control_indices', None)
            if isinstance(_ci, dict) and isinstance(_ci.get(pname), int):
                return _ci[pname]
            return getattr(p, 'control_scenario', 0)

        def _get_sol_keys(pname):
            _first_sol = sd[f"sensitivity_of_solutions_to_{pname}"][_control_idx_for(pname)]
            if isinstance(_first_sol, list):
                # multi-objective: one dict per Pareto point, all sharing the
                # same variable names, so the first point is representative
                _first_sol = next((s for s in _first_sol
                                   if isinstance(s, dict)), None)
            if not isinstance(_first_sol, dict):
                return []
            _effective_sol = _first_sol
            if not _is_single:
                for _pk, _pv in _first_sol.items():
                    if isinstance(_pv, dict):
                        _effective_sol = _pv
                        break
            keys = []
            # Only return keys if key_vars is explicitly provided (not empty)
            if len(p.key_vars) == 0:
                return []  # No key_vars specified, don't show any variables
            for k in _effective_sol:
                if not isinstance(k, str):
                    continue
                if hidden_variables or (not k.startswith('_') and 'autolin' not in k and 'sos2' not in k and 'rdiv' not in k):
                    if k in p.key_vars:
                        keys.append(k)
            return keys

        def _get_term_names(pname):
            # Named terms registered with m.term(name, value): shown as
            # percent-change columns right after CPT, before key
            # variables, in registration order.
            try:
                _tf = (p.em.features if getattr(p, 'em', None) is not None else {}) or {}
                _imm = _tf.get('_terms') or {}
                _dfr = _tf.get('_deferred_terms') or {}
                _order = _tf.get('_term_order')
                if _order:
                    _tn = [k for k in _order if k in _imm or k in _dfr]
                else:
                    _tn = list(_imm.keys()) + [k for k in _dfr.keys() if k not in _imm]
            except Exception:
                return []
            if not _tn:
                return []
            # keep only terms that actually landed in the recorded solutions
            try:
                _sol_list = sd[f"sensitivity_of_solutions_to_{pname}"]
                _ci = _control_idx_for(pname)
                _sol = _sol_list[_ci] if 0 <= _ci < len(_sol_list) else None
                if isinstance(_sol, list):
                    # multi-objective: one dict per Pareto point, every point
                    # carrying the same m.term() names
                    _tn = [k for k in _tn
                           if any(isinstance(_p, dict) and k in _p
                                  for _p in _sol)]
                    _sol = None
                if not (isinstance(_sol, dict) and _sol):
                    # control scenario failed/not recorded: keep terms that
                    # landed in any recorded scenario instead of dropping
                    # the whole column (per-row cells still show '-')
                    _sol = next((s for s in _sol_list
                                 if isinstance(s, dict) and s), None)
                if isinstance(_sol, dict):
                    _eff = _sol
                    if not _is_single:
                        for _pk, _pv in _sol.items():
                            if isinstance(_pv, dict):
                                _eff = _pv
                                break
                    _tn = [k for k in _tn if k in _eff]
            except Exception:
                pass
            return _tn

        _names = list(p.sensitivity_parameter_names)
        _pvals = list(p.sensitivity_parameter_values)
        _control_indices = getattr(p, 'sensitivity_control_indices', None)
        _default_base_idx = getattr(p, 'control_scenario', 0)

        _filtered_names = []
        _filtered_pvals = []
        for _fi, _fn in enumerate(_names):
            _fv = _pvals[_fi] if _fi < len(_pvals) else []
            if _fv and isinstance(_fv[0], set):
                continue
            _filtered_names.append(_fn)
            _filtered_pvals.append(_fv)
        _names = _filtered_names
        _pvals = _filtered_pvals

        _param_info = []
        _all_obj_vals = []

        for pi, pname in enumerate(_names):
            pvals = _pvals[pi]
            if _control_indices and pname in _control_indices:
                bi = _control_indices[pname]
            else:
                bi = _default_base_idx if _default_base_idx < len(pvals) else 0
            base_obj = sd[f"sensitivity_of_objectives_to_{pname}"][bi]
            base_val = pvals[bi]

            obj_vals = []
            for v in sd[f"sensitivity_of_objectives_to_{pname}"]:
                if v is not None:
                    agg = _aggregate_objective(v)
                    if agg is not None:
                        obj_vals.append(agg)

            _all_obj_vals.append(obj_vals)

            if obj_vals and base_obj is not None:
                base_f = _aggregate_objective(base_obj)
                if base_f is None:
                    base_f = 0.0
                lo = min(obj_vals)
                hi = max(obj_vals)
                obj_std = float(_np.std(obj_vals)) if len(obj_vals) > 1 else 0.0
                obj_range = hi - lo

                if base_f != 0:
                    lo_pct = (lo - base_f) / abs(base_f) * 100
                    hi_pct = (hi - base_f) / abs(base_f) * 100
                    impact = abs(hi_pct - lo_pct)
                    normalized_sensitivity = obj_std / abs(base_f) if base_f != 0 else 0
                else:
                    lo_pct = hi_pct = impact = normalized_sensitivity = 0

                if _obj_dir == 'min':
                    best_idx, worst_idx = obj_vals.index(lo), obj_vals.index(hi)
                    best_obj, worst_obj = lo, hi
                else:
                    best_idx, worst_idx = obj_vals.index(hi), obj_vals.index(lo)
                    best_obj, worst_obj = hi, lo
            else:
                lo = hi = base_f = lo_pct = hi_pct = impact = normalized_sensitivity = 0
                best_idx = worst_idx = bi
                best_obj = worst_obj = base_f

            _param_info.append({
                'name': pname,
                'desc': _descriptions.get(pname, ''),
                'base_val': base_val,
                'base_obj': base_obj,
                'best_obj': best_obj, 'worst_obj': worst_obj,
                'best_idx': best_idx, 'worst_idx': worst_idx,
                'lo_pct': lo_pct, 'hi_pct': hi_pct,
                'impact': impact,
                'normalized_sensitivity': normalized_sensitivity,
                'n_scenarios': len(pvals),
            })

        if _all_obj_vals:
            _all_flat = [v for sublist in _all_obj_vals for v in sublist]
            _total_var = float(_np.var(_all_flat)) if _all_flat else 0
            for _ci, info in enumerate(_param_info):
                _obj_variance = float(_np.var(_all_obj_vals[_ci])) if _ci < len(_all_obj_vals) and _all_obj_vals[_ci] else 0
                info['variance_contribution'] = _obj_variance / _total_var if _total_var > 0 else 0
        else:
            for info in _param_info:
                info['variance_contribution'] = 0

        _param_info.sort(key=lambda x: x['impact'], reverse=True)
        _info_by_name = {info['name']: info for info in _param_info}

        def _ra(text):
            _vw = box._visible_len(text)
            _pad = max(0, box.width - 4 - _vw)
            return " " * _pad + text

        _unified_val_w = 13
        _unified_obj_w = 7
        _unified_pct_w = 6
        _unified_pval_pct_w = 6
        _unified_sol_ws = {}

        for pi, pname in enumerate(_names):
            pvals = _pvals[pi]
            _sol_keys = _get_sol_keys(pname)
            info = _info_by_name.get(pname)
            _base_obj = info['base_obj'] if info else None
            if _control_indices and pname in _control_indices:
                _bi = _control_indices[pname]
            else:
                _bi = _default_base_idx if _default_base_idx < len(pvals) else 0
            _base_cpt = sd[f"sensitivity_of_cpt_to_{pname}"][_bi] if f"sensitivity_of_cpt_to_{pname}" in sd else None
            for v in pvals:
                _w = max(13, len(_format_base_val(v)) + 1)
                if _w > _unified_val_w:
                    _unified_val_w = _w
            _base_pval = pvals[bi] if bi < len(pvals) else None
            for v in pvals:
                _pw = len(_pval_pct_str(v, _base_pval))
                if _pw > _unified_pval_pct_w:
                    _unified_pval_pct_w = _pw
            for _oi, _ov in enumerate(sd.get(f"sensitivity_of_objectives_to_{pname}", [])):
                if _ov is not None and _base_obj is not None:
                    _agg_ov = _aggregate_objective(_ov)
                    _agg_base = _aggregate_objective(_base_obj)
                    _pw = len(_pct_str(_agg_ov, _agg_base))
                    if _pw > _unified_pct_w:
                        _unified_pct_w = _pw
            for _ci, _cv in enumerate(sd.get(f"sensitivity_of_cpt_to_{pname}", [])):
                if _cv is not None and _base_cpt is not None:
                    _pw = len(_pct_str(_cv, _base_cpt))
                    if _pw > _unified_pct_w:
                        _unified_pct_w = _pw
            for sk in _sol_keys:
                _klen = len(sk)
                _sk_vtype = _var_types.get(sk, None)
                for _si in range(len(pvals)):
                    _sv = sd.get(f"sensitivity_of_solutions_to_{pname}", [{}])[_si]
                    if isinstance(_sv, dict) and sk in _sv:
                        _svl = len(_summarize_var(_sv[sk], _sk_vtype))
                        if _svl > _klen:
                            _klen = _svl
                _klen = max(_klen, 3)
                if sk not in _unified_sol_ws or _klen > _unified_sol_ws[sk]:
                    _unified_sol_ws[sk] = _klen

        _delta_L = '\u03b4'

        _dataset = getattr(p, 'inputdata', None)
        _std_params = getattr(_dataset, 'std_params', {}) if _dataset else {}
        _avg_params = getattr(_dataset, 'average_params', {}) if _dataset else {}
        _min_params = getattr(_dataset, 'minimum_params', {}) if _dataset else {}
        _max_params = getattr(_dataset, 'maximum_params', {}) if _dataset else {}

        def _build_param_inner(pname, pvals, info, bi, sd, _is_single, col_w, sol_keys=None, term_names=None, n_obj=None):
            """
            Build inner content lines for a single parameter column (no borders, no header).
            Now includes decision variable changes alongside Obj/CPT.
            """
            base_obj = info['base_obj'] if info else None
            base_cpt = sd[f"sensitivity_of_cpt_to_{pname}"][bi] if f"sensitivity_of_cpt_to_{pname}" in sd else None
            sol_data = sd.get(f"sensitivity_of_solutions_to_{pname}", [])
            
            # Get key variables if available
            _key_vars = sol_keys if sol_keys else []
            # m.term() names → percent-change columns after CPT, before vars
            _terms = term_names if term_names else []
            # one Obj column per objective: a multi-objective run has no
            # single objective to quote, so each column is the percent change
            # of THAT objective's average across the Pareto front
            _n_o = 1 if _is_single else max(1, int(n_obj or 1))

            def _obj_pct_cells(curr_v, base_v):
                if _n_o <= 1:
                    return [_pct_str(_aggregate_objective(curr_v),
                                     _aggregate_objective(base_v))]
                _cm = ReportEngine._objective_means(curr_v, _n_o)
                _bm = ReportEngine._objective_means(base_v, _n_o)
                _cells = []
                for _k in range(_n_o):
                    _cv = _cm[_k] if _cm is not None and _k < len(_cm) else None
                    _bv = _bm[_k] if _bm is not None and _k < len(_bm) else None
                    _cells.append(_pct_str(_cv, _bv))
                return _cells

            def _pct6(curr_v, base_v):
                # percent change of a term value vs the control scenario,
                # fitted into 6 chars (same convention as variable cells)
                try:
                    _bv = float(_np.mean(_np.asarray(base_v, dtype=float)))
                    _cv = float(_np.mean(_np.asarray(curr_v, dtype=float)))
                    if abs(_bv) <= 1e-10:
                        return "-"
                    _tp = (_cv - _bv) / abs(_bv) * 100
                except Exception:
                    return "-"
                for _dec in (1, 0):
                    _t = f"{_tp:+.{_dec}f}"
                    if len(_t) <= 6:
                        return _t
                for _dec in (1, 0):
                    _t = f"{_tp:+.{_dec}e}"
                    if len(_t) <= 6:
                        return _t
                return f"{_tp:+.0e}"[:6]

            def _as_avg_sol(sol):
                """Recorded solution as one flat dict of per-front averages.

                A multi-objective run records one dict per Pareto point -- as
                a list, or as a mapping keyed by point index once the
                sensitivity sweep has stored it -- so the points are averaged
                here, the same summary the Obj columns use.  A plain
                single-objective dict (whose values are variable payloads,
                keyed by name) is passed straight through.
                """
                if isinstance(sol, dict):
                    _vals = list(sol.values())
                    _pareto_map = (
                        bool(_vals)
                        and all(isinstance(v, dict) for v in _vals)
                        and all(isinstance(k, int) for k in sol))
                    if not _pareto_map:
                        return sol
                    sol = _vals
                if not isinstance(sol, list):
                    return None
                _buckets = {}
                for _pt in sol:
                    if isinstance(_pt, dict):
                        for _k, _v in _pt.items():
                            _buckets.setdefault(_k, []).append(_v)
                if not _buckets:
                    return None
                _agg = {}
                for _k, _vs in _buckets.items():
                    try:
                        _agg[_k] = float(_np.mean(_np.asarray(_vs, dtype=float)))
                    except (TypeError, ValueError):
                        _agg[_k] = _vs[0]
                return _agg

            def _sort_key(idx):
                pv = pvals[idx]
                if isinstance(pv, (_np.ndarray, list, tuple)):
                    _arr = _np.asarray(pv, dtype=float)
                    if _arr.ndim == 2 and _arr.shape[1] == 2:
                        _centroid = _np.mean(_arr, axis=0)
                        _spread = float(_np.mean(_np.abs(_arr - _centroid)))
                        return (-_arr.shape[0], -_spread)
                    return float(_np.mean(_arr))
                try:
                    return float(pv)
                except (TypeError, ValueError):
                    return 0.0

            _sorted_indices = sorted(range(len(pvals)), key=_sort_key)
            if bi in _sorted_indices:
                _sorted_indices.remove(bi)
                _sorted_after = [i for i in _sorted_indices if _sort_key(i) > _sort_key(bi)]
                _sorted_before = [i for i in _sorted_indices if _sort_key(i) <= _sort_key(bi)]
                if _sorted_after and not _sorted_before:
                    _sorted_indices = [bi] + _sorted_indices
                elif not _sorted_after and _sorted_before:
                    _sorted_indices = _sorted_indices + [bi]
                else:
                    _mid = len(_sorted_indices) // 2
                    _sorted_indices.insert(_mid, bi)

            lines = []
            # Build header with decision variables
            if _n_o > 1:
                _obj_headers = "".join(
                    f" {'Obj%d' % (_k + 1):>6}" for _k in range(_n_o))
            else:
                _obj_headers = f" {'Obj':>6}"
            _term_headers = ""
            if _terms:
                # names >6 chars keep an ellipsis (Name-column convention)
                _term_headers = "".join(
                    f" {(tk if len(tk) <= 6 else tk[:3] + '...'):>6}" for tk in _terms)
            if _key_vars or _terms:
                # Include variable names in header
                var_headers = "".join(f" {vk:>6}" for vk in _key_vars[:3])  # Limit to 3 vars max
                hdr = f" {'Val':>6}{_obj_headers} {'CPT':>6}{_term_headers}{var_headers}"
            else:
                hdr = f" {'Val':>6}{_obj_headers} {'CPT':>6}"
            
            hdr_vw = box._visible_len(hdr)
            if hdr_vw < col_w:
                hdr = hdr + " " * (col_w - hdr_vw)
            elif hdr_vw > col_w:
                hdr = hdr[:col_w]
            lines.append(hdr)

            # Build data rows
            health_data = sd.get(f"sensitivity_of_health_to_{pname}", [])
            for i in _sorted_indices:
                obj_val = sd[f"sensitivity_of_objectives_to_{pname}"][i]
                cpt_val = sd[f"sensitivity_of_cpt_to_{pname}"][i] if f"sensitivity_of_cpt_to_{pname}" in sd else None
                pval = pvals[i]
                _pval_pct = _pval_pct_str(pval, pvals[bi])
                # Detect failed (build/run raised) vs infeasible scenarios
                _is_failed = False
                _is_infeasible = False
                if i < len(health_data):
                    if health_data[i] is None:
                        _is_failed = True
                    elif health_data[i] is False:
                        _is_infeasible = True
                if obj_val is None and i != bi and not _is_failed:
                    _is_infeasible = True
                _obj_cells = _obj_pct_cells(obj_val, base_obj)
                _cpt_pct = _pct_str(cpt_val, base_cpt)

                # recorded solution: a dict for single objective, one dict
                # per Pareto point for multi-objective (averaged to one)
                _base_sol = (_as_avg_sol(sol_data[bi])
                             if bi < len(sol_data) else None)
                _curr_sol = (_as_avg_sol(sol_data[i])
                             if i < len(sol_data) else None)

                # Term values (m.term): percent change vs control scenario
                term_values = ""
                if _terms:
                    if isinstance(_base_sol, dict) and isinstance(_curr_sol, dict):
                        for tk in _terms:
                            if tk in _base_sol and tk in _curr_sol:
                                term_values += f" {_pct6(_curr_sol[tk], _base_sol[tk]):>6}"
                            else:
                                term_values += f" {'-':>6}"
                    else:
                        for _ in _terms:
                            term_values += f" {'-':>6}"

                # Add decision variable values
                var_values = ""
                if _key_vars and i < len(sol_data) and bi < len(sol_data):
                    base_sol = _base_sol
                    curr_sol = _curr_sol
                    if isinstance(base_sol, dict) and isinstance(curr_sol, dict):
                        for vk in _key_vars[:3]:
                            if vk in base_sol and vk in curr_sol:
                                try:
                                    base_v = float(_np.mean(_np.asarray(base_sol[vk], dtype=float)))
                                    curr_v = float(_np.mean(_np.asarray(curr_sol[vk], dtype=float)))
                                    if abs(base_v) > 1e-10:
                                        var_pct = (curr_v - base_v) / abs(base_v) * 100
                                        vs = None
                                        for dec in (1, 0):
                                            t = f"{var_pct:+.{dec}f}"
                                            if len(t) <= 6:
                                                vs = t
                                                break
                                        if vs is None:
                                            for dec in (1, 0):
                                                t = f"{var_pct:+.{dec}e}"
                                                if len(t) <= 6:
                                                    vs = t
                                                    break
                                        if vs is None:
                                            vs = f"{var_pct:+.0e}"
                                        var_values += f" {vs[:6]:>6}"
                                    else:
                                        var_values += f" {'-':>6}"
                                except Exception:
                                    var_values += f" {'-':>6}"
                            else:
                                var_values += f" {'-':>6}"
                
                if _is_failed:
                    _obj_cells = ["FAIL"] + ["-"] * (_n_o - 1)
                    _cpt_pct = "-"
                    term_values = "".join(f" {'-':>6}" for _ in _terms)
                    var_values = ""
                    for _ in range(min(3, len(_key_vars))):
                        var_values += f" {'-':>6}"
                elif _is_infeasible:
                    _obj_cells = ["INFEAS"] + ["-"] * (_n_o - 1)
                    _cpt_pct = "-"
                    term_values = "".join(f" {'-':>6}" for _ in _terms)
                    var_values = ""
                    for _ in range(min(3, len(_key_vars))):
                        var_values += f" {'-':>6}"

                row = (f" {_pval_pct:>6}"
                       + "".join(f" {_c:>6}" for _c in _obj_cells)
                       + f" {_cpt_pct:>6}{term_values}{var_values}")
                rvw = box._visible_len(row)
                if rvw < col_w:
                    row = row + " " * (col_w - rvw)
                elif rvw > col_w:
                    row = row[:col_w]
                lines.append(row)

            return lines

        def _build_impact_footer(_impact_entry, col_w, pname, _interaction_data, health_data=None):
            """Build footer lines with sensitivity metrics and interactions."""
            import math as _math
            lines = []
            lines.append(" " * col_w)
            _sj = _impact_entry['sensitivity']
            _dj = _impact_entry['decision']
            _uj = _impact_entry['uncertainty']
            _cpt_s = _impact_entry['cpt_sensitivity']
            _aj = _impact_entry['asymmetry']
            _ej = _impact_entry['weighted_impact']

            def _fmt(v):
                if v is None or (_math.isnan(v) if isinstance(v, float) else False):
                    return f"{'-':>6}"
                if _math.isinf(v) if isinstance(v, float) else False:
                    return f"{'∞':>6}" if v > 0 else f"{'-∞':>6}"
                s = f"{v:.2f}"
                return s.rjust(6) if len(s) <= 6 else s[:6]

            # Count failed (exception) and infeasible scenarios
            _n_infeas = 0
            _n_fail = 0
            if health_data:
                _n_infeas = sum(1 for h in health_data if h is False)
                _n_fail = sum(1 for h in health_data if h is None)

            # Sensitivity metrics (existing)
            frow1 = f" {'Sen':>6} {'Dec':>6} {'Unc':>6}"
            frow2 = f" {_fmt(_sj)} {_fmt(_dj)} {_fmt(_uj)}"
            frow3 = f" {'Tim':>6} {'Asy':>6} {'Imp':>6}"
            frow4 = f" {_fmt(_cpt_s)} {_fmt(_aj)} {_fmt(_ej)}"
            for fr in [frow1, frow2, frow3, frow4]:
                fvw = box._visible_len(fr)
                if fvw < col_w:
                    fr = fr + " " * (col_w - fvw)
                elif fvw > col_w:
                    fr = fr[:col_w]
                lines.append(fr)

            # Infeasibility summary line
            if _n_infeas > 0:
                _total = len(health_data) if health_data else 0
                _inf_line = f" Infeas: {_n_infeas}/{_total} scenarios"
                fvw = box._visible_len(_inf_line)
                if fvw < col_w:
                    _inf_line = _inf_line + " " * (col_w - fvw)
                elif fvw > col_w:
                    _inf_line = _inf_line[:col_w]
                lines.append(_inf_line)
            if _n_fail > 0:
                _total = len(health_data) if health_data else 0
                _fail_line = f" Failed: {_n_fail}/{_total} scenarios"
                fvw = box._visible_len(_fail_line)
                if fvw < col_w:
                    _fail_line = _fail_line + " " * (col_w - fvw)
                elif fvw > col_w:
                    _fail_line = _fail_line[:col_w]
                lines.append(_fail_line)
            
            return lines

        def _pad_to_height(lines, height, col_w):
            while len(lines) < height:
                lines.append(" " * col_w)
            return lines

        # Get interaction data early to include in sensitivity boxes
        _interaction_data = getattr(p, 'sensitivity_interaction_data', [])
        
        # Build interaction data structure for easy lookup
        _int_rows = []
        if _interaction_data:
            for ix in _interaction_data:
                _pair = ix['pair']
                _r_ab = 0.0
                _i_type = "Neutral"
                _nonlin = 0.0
                
                if ix['i_ab'] is not None and ix['z00'] is not None and ix['z10'] is not None and ix['z01'] is not None:
                    _main_a = abs(ix['z10'] - ix['z00'])
                    _main_b = abs(ix['z01'] - ix['z00'])
                    _denom = _main_a + _main_b
                    _r_ab = abs(ix['i_ab']) / _denom if _denom > 1e-10 else 0.0
                    
                    if ix['i_ab'] > 1e-10:
                        _i_type = "Synergy"
                    elif ix['i_ab'] < -1e-10:
                        _i_type = "Conflict"
                
                if ix.get('z00') is not None and ix.get('z11') is not None:
                    _total_effect = abs(ix['z11'] - ix['z00'])
                    if _total_effect > 1e-10 and ix.get('i_ab') is not None:
                        _nonlin = min(abs(ix['i_ab']) / _total_effect, 1.0)
                
                _int_rows.append({
                    'pair': _pair,
                    'type': _i_type,
                    'term': ix['i_ab'],
                    'strength': _r_ab,
                    'nonlin': _nonlin,
                    'z00': ix['z00'],
                    'z10': ix['z10'],
                    'z01': ix['z01'],
                    'z11': ix['z11'],
                    'd_10': ix['d_10'],
                    'd_01': ix['d_01'],
                    'd_11': ix['d_11']
                })
        
        # Get key variables to determine column layout
        _sol_keys = _get_sol_keys(_names[0]) if _names else []
        _n_vars = len(_sol_keys)
        _var_overhead = _n_vars * 7 if _n_vars > 0 else 0  # 7 chars per var col
        # m.term() terms get their own columns (after CPT, before vars)
        _term_names = _get_term_names(_names[0]) if _names else []
        _term_overhead = len(_term_names) * 7 if _term_names else 0  # 7 chars per term col
        # a multi-objective problem gets one Obj column per objective, each
        # holding that objective's own percent change
        _n_obj = int(getattr(p, 'number_of_objectives', 1) or 1)
        _n_obj_cols = 1 if _is_single else max(1, _n_obj)
        _extra_cols = _n_vars + len(_term_names) + (_n_obj_cols - 1)

        inner_width = box.width - 4
        sep = " │ "
        sep_w = len(sep)

        # Adaptive column layout: pick max cols that fit base+vars+terms
        _base_min_w = 7 * (2 + _n_obj_cols)  # Val + one col/objective + CPT
        _min_needed = _base_min_w + _var_overhead + _term_overhead
        if _extra_cols <= 1:
            _n_cols = 3
        elif _extra_cols <= 2:
            _n_cols = 2
        else:
            _n_cols = 1
        # Shrink columns if vars make them too narrow
        while _n_cols > 1:
            _cw = (inner_width - sep_w * (_n_cols - 1)) // _n_cols
            if _cw >= _min_needed:
                break
            _n_cols -= 1

        # Distribute extra space evenly across columns
        base_col_w = (inner_width - sep_w * (_n_cols - 1)) // _n_cols
        extra_space = (inner_width - sep_w * (_n_cols - 1)) % _n_cols
        col_widths = [base_col_w + (1 if i < extra_space else 0) for i in range(_n_cols)]
        col_w = base_col_w  # Keep for backward compatibility

        _param_cols = []
        _impact_by_name = {}
        for pi, pname in enumerate(_names):
            pvals = _pvals[pi]
            info = _info_by_name.get(pname)
            if _control_indices and pname in _control_indices:
                bi = _control_indices[pname]
            else:
                bi = _default_base_idx if _default_base_idx < len(pvals) else 0
            _impact_entry = self.compute_impact(pname, pvals, bi, sd)
            _impact_by_name[pname] = _impact_entry
            # Use the appropriate column width for this position
            _col_idx = pi % _n_cols
            _col_w = col_widths[_col_idx]
            _col = _build_param_inner(pname, pvals, info, bi, sd, _is_single, _col_w, sol_keys=_sol_keys, term_names=_term_names, n_obj=_n_obj_cols)
            _param_cols.append(_col)

        _footer_lengths = []
        for i, pname in enumerate(_names):
            if pname in _impact_by_name:
                _col_idx = i % _n_cols
                _col_w = col_widths[_col_idx]
                _health = sd.get(f"sensitivity_of_health_to_{pname}", [])
                _footer = _build_impact_footer(_impact_by_name[pname], _col_w, pname, _int_rows, health_data=_health)
                _footer_lengths.append(len(_footer))
                _param_cols[i].extend(_footer)
            else:
                _footer_lengths.append(0)

        has_cols = len(_param_cols) > 0

        # Calculate separator positions using actual column widths
        sep_positions = []
        for ci in range(_n_cols - 1):
            # Position: border(1) + space(1) + columns_before + seps_before + middle_of_sep(1)
            cols_before_width = sum(col_widths[:ci+1])
            pos = 1 + 1 + cols_before_width + ci * sep_w + 1
            sep_positions.append(pos)

        def _build_border_line(left_ch, bar_ch, mid_ch, right_ch, label=""):
            line = [left_ch] + [bar_ch] * (box.width - 2) + [right_ch]
            for sp in sep_positions:
                if 0 < sp < len(line):
                    line[sp] = mid_ch
            if label:
                label_str = f" {label} "
                for i, ch in enumerate(label_str):
                    if 1 + i < len(line) - 1:
                        line[1 + i] = ch
            return "".join(line)

        # _rows_of_cols uses the _n_cols already determined above (adaptive based on variables)
        _rows_of_cols = []
        for start in range(0, len(_param_cols), _n_cols):
            _rows_of_cols.append(_param_cols[start:start + _n_cols])

        for _row_idx, _row_cols in enumerate(_rows_of_cols):
            _actual_n = len(_row_cols)

            # Only pad columns to equal height when multiple boxes share a row
            if _actual_n > 1:
                _data_parts = []
                _foot_parts = []
                _foot_lens = []
                for _ri in range(_actual_n):
                    _r_ci = _row_idx * _n_cols + _ri
                    _fl = _footer_lengths[_r_ci] if _r_ci < len(_footer_lengths) else 0
                    _foot_lens.append(_fl)
                    if _fl > 0 and len(_row_cols[_ri]) >= _fl:
                        _data_parts.append(_row_cols[_ri][:-_fl])
                        _foot_parts.append(_row_cols[_ri][-_fl:])
                    else:
                        _data_parts.append(_row_cols[_ri])
                        _foot_parts.append([])
                _row_max_h = max(len(d) for d in _data_parts) if _data_parts else 0
                for _ri in range(_actual_n):
                    _r_ci = _row_idx * _n_cols + _ri
                    _r_cw = col_widths[_r_ci % _n_cols]
                    _data_parts[_ri] = _pad_to_height(_data_parts[_ri], _row_max_h, _r_cw)
                    _row_cols[_ri] = _data_parts[_ri] + _foot_parts[_ri]

            # Calculate separator positions in data rows
            # Row structure: border(1) + " "(1) + col + sep + col + sep + ... + " "(1) + border(1)
            # The middle of sep " │ " is at offset 1 within sep (the │ char)
            _row_sep_positions = []
            for ci in range(_actual_n - 1):
                # Position: border(1) + space(1) + columns_before + seps_before + position_in_sep(1 for │)
                cols_before_width = sum(col_widths[:ci+1])
                pos = 1 + 1 + cols_before_width + ci * sep_w + 1
                _row_sep_positions.append(pos)

            def _build_row_border(left_ch, bar_ch, mid_ch, right_ch, label=""):
                line = [left_ch] + [bar_ch] * (box.width - 2) + [right_ch]
                for sp in _row_sep_positions:
                    if 0 < sp < len(line):
                        line[sp] = mid_ch
                if label:
                    label_str = f" {label} "
                    for i, ch in enumerate(label_str):
                        if 1 + i < len(line) - 1:
                            line[1 + i] = ch
                return "".join(line)

            _lbl = " Sensitivity " if _row_idx == 0 else ""
            top_line = _build_row_border(box._topleft, box._topbar, box._middletop, box._topright, label=_lbl)
            box._emit(top_line, "top")

            _row_names = _names[_row_idx * _n_cols:(_row_idx + 1) * _n_cols]
            hdr_cells = []
            for col_idx, pname in enumerate(_row_names):
                _desc = _info_by_name[pname]['desc'] if pname in _info_by_name and _info_by_name[pname] else ''
                _eff = _impact_by_name.get(pname, {}).get('direction_effect', '')
                # Use the actual column width for this column position
                _actual_col_idx = _row_idx * _n_cols + col_idx
                _col_w = col_widths[_actual_col_idx % _n_cols]
                # " <name>" ... "<glyph> <word>", marker flush right so the box
                # reads without consulting the legend.  Narrow columns fall back
                # to the bare glyph, then to the name alone.
                _name_part = f" {pname}"
                _nlen = box._visible_len(_name_part)
                _tag = f"{_eff} {DIRECTION_HINTS[_eff]}" if _eff in DIRECTION_HINTS else _eff
                _label = _name_part
                for _try in (_tag, _eff):
                    if _try and _nlen + box._visible_len(_try) + 1 <= _col_w:
                        _label = _name_part + " " * (_col_w - _nlen - box._visible_len(_try)) + _try
                        break
                hdr_cells.append(_label.ljust(_col_w)[:_col_w])
            hdr_inner = sep.join(hdr_cells)
            pad = inner_width - box._visible_len(hdr_inner)
            if pad > 0:
                hdr_inner += " " * pad
            box._emit(f"{box._border} {hdr_inner} {box._border}", "row")

            _row_max_h = max(len(c) for c in _row_cols) if _row_cols else 0
            for ci_idx in range(len(_row_cols)):
                _actual_col_idx = _row_idx * _n_cols + ci_idx
                _col_w = col_widths[_actual_col_idx % _n_cols]
                _row_cols[ci_idx] = _pad_to_height(_row_cols[ci_idx], _row_max_h, _col_w)

            for row_idx in range(_row_max_h):
                cells = []
                for ci in range(_actual_n):
                    cells.append(_row_cols[ci][row_idx])
                row_inner = sep.join(cells)
                pad = inner_width - box._visible_len(row_inner)
                if pad > 0:
                    row_inner += " " * pad
                box._emit(f"{box._border} {row_inner} {box._border}", "row")

            # Bottom line: use ┴ if this is the last row, otherwise ┼
            is_last_row = (_row_idx == len(_rows_of_cols) - 1)
            if is_last_row:
                # Check if there are interactions following
                _has_interactions = bool(_int_rows)
                if _has_interactions:
                    # Use ┼ at separator positions and ├/┤ at edges since interactions follow
                    bot_line = _build_row_border("\u251c", box._bottombar, "\u253c", "\u2524")  # ├─┼─┤
                    box._emit(bot_line, "bottom")
                else:
                    # Use ┴ at separator positions (normal bottom) - will be emitted below with timing
                    bot_line = _build_row_border(box._bottomleft, box._bottombar, box._middlebottom, box._bottomright)

                # --- Interaction table (unique pairs, same column layout as sensitivity) ---
                if _int_rows:
                    # Use same number of columns as sensitivity boxes
                    _int_n_cols = _n_cols
                    _int_col_widths = col_widths[:_int_n_cols]
                    
                    # Calculate separator positions for interaction section
                    _int_sep_positions = []
                    for ci in range(_int_n_cols - 1):
                        cols_before_width = sum(_int_col_widths[:ci+1])
                        pos = 1 + 1 + cols_before_width + ci * sep_w + 1
                        _int_sep_positions.append(pos)
                    
                    # Build border function with proper separator positions
                    def _build_int_border(left_ch, bar_ch, mid_ch, right_ch):
                        line = [left_ch] + [bar_ch] * (box.width - 2) + [right_ch]
                        for sp in _int_sep_positions:
                            if 0 < sp < len(line):
                                line[sp] = mid_ch
                        return "".join(line)
                    
                    # Don't emit a separate top border - sensitivity bottom already serves as separator

                    # Data rows - each pair gets its own header, sub-header, and values
                    for _row_start in range(0, len(_int_rows), _int_n_cols):
                        _row_pairs = _int_rows[_row_start:_row_start + _int_n_cols]
                        _n_pairs_in_row = len(_row_pairs)

                        # Pair name row (header)
                        _name_cells = []
                        for _ri in range(_int_n_cols):
                            _cw = _int_col_widths[_ri]
                            if _ri < _n_pairs_in_row:
                                _rp = _row_pairs[_ri]
                                # " <pair>" ... "<Synergy|Conflict|Neutral>" with
                                # the label flush right, mirroring the way the
                                # parameter boxes show their marker.
                                _np_str = f" {_rp['pair']}"
                                _rp_type = _rp.get('type') or ''
                                if _rp_type:
                                    _npvw = box._visible_len(_np_str)
                                    _tpvw = box._visible_len(_rp_type)
                                    if _npvw + _tpvw + 1 <= _cw:
                                        _np_str = _np_str + " " * (_cw - _npvw - _tpvw) + _rp_type
                                    else:
                                        # Too narrow: keep the label, trim the pair
                                        # name with the usual "..." convention.
                                        _room = _cw - _tpvw - 2
                                        if _room >= 5:
                                            _disp = _rp['pair']
                                            if box._visible_len(_disp) > _room:
                                                _disp = _disp[:_room - 3] + "..."
                                            _np_str = (f" {_disp}"
                                                       + " " * (_cw - 1 - box._visible_len(_disp) - _tpvw)
                                                       + _rp_type)
                                _npvw = box._visible_len(_np_str)
                                if _npvw < _cw:
                                    _np_str = _np_str.ljust(_cw)
                                elif _npvw > _cw:
                                    _np_str = _np_str[:_cw]
                                _name_cells.append(_np_str)
                            else:
                                # Empty cell for missing interaction
                                _name_cells.append(" " * _cw)
                        _name_inner = sep.join(_name_cells)
                        _name_pad = inner_width - box._visible_len(_name_inner)
                        if _name_pad > 0:
                            _name_inner += " " * _name_pad
                        box._emit(f"{box._border} {_name_inner} {box._border}", "row")

                        # Sub-header row (align with Sen/Dec/Unc format above)
                        _int_sub_cells = []
                        for _ci in range(_int_n_cols):
                            _cw = _int_col_widths[_ci]
                            if _ci < _n_pairs_in_row:
                                # Match the spacing of Sen/Dec/Unc: " {val:>6} {val:>6} {val:>6}"
                                _s = f" {'Str':>6} {'Dec':>6} {'Non':>6}"
                                _svw = box._visible_len(_s)
                                if _svw < _cw:
                                    _s = _s + " " * (_cw - _svw)
                                elif _svw > _cw:
                                    _s = _s[:_cw]
                                _int_sub_cells.append(_s)
                            else:
                                # Empty cell
                                _int_sub_cells.append(" " * _cw)
                        _int_sub_inner = sep.join(_int_sub_cells)
                        _sub_pad = inner_width - box._visible_len(_int_sub_inner)
                        if _sub_pad > 0:
                            _int_sub_inner += " " * _sub_pad
                        box._emit(f"{box._border} {_int_sub_inner} {box._border}", "row")

                        # Values row (align with Sen/Dec/Unc values above)
                        _val_cells = []
                        for _ri in range(_int_n_cols):
                            _cw = _int_col_widths[_ri]
                            if _ri < _n_pairs_in_row:
                                _rp = _row_pairs[_ri]
                                # Match the spacing: " {val:>6.2f} {val:>6.2f} {val:>6.2f}"
                                _v = f" {_rp['strength']:>6.2f} {_rp.get('d_11', 0.0):>6.2f} {_rp.get('nonlin', 0.0):>6.2f}"
                                _vvw = box._visible_len(_v)
                                if _vvw < _cw:
                                    _v = _v + " " * (_cw - _vvw)
                                elif _vvw > _cw:
                                    _v = _v[:_cw]
                                _val_cells.append(_v)
                            else:
                                # Empty cell
                                _val_cells.append(" " * _cw)
                        _val_inner = sep.join(_val_cells)
                        _val_pad = inner_width - box._visible_len(_val_inner)
                        if _val_pad > 0:
                            _val_inner += " " * _val_pad
                        box._emit(f"{box._border} {_val_inner} {box._border}", "row")

                    # Bottom line with ┴ separators
                    elapsed = p.sensitivity_end_timer - p.sensitivity_begin_timer
                    _time_str = format_time_duration(elapsed, name="")
                    _int_bot = _build_int_border(box._bottomleft, box._bottombar, box._middlebottom, box._bottomright)
                    if _time_str:
                        _time_with_space = f" {_time_str} "
                        _time_len = len(_time_with_space)
                        if len(_int_bot) >= _time_len + 2:
                            _int_bot = _int_bot[:-_time_len-1] + _time_with_space + box._bottomright
                    box._emit(_int_bot, "bottom")
                else:
                    # No interactions, just add time to bottom
                    elapsed = p.sensitivity_end_timer - p.sensitivity_begin_timer
                    _time_str = format_time_duration(elapsed, name="")
                    if _time_str:
                        _time_with_space = f" {_time_str} "
                        _time_len = len(_time_with_space)
                        if len(bot_line) >= _time_len + 2:
                            bot_line = bot_line[:-_time_len-1] + _time_with_space + box._bottomright
                    box._emit(bot_line, "bottom")
                return  # Skip the standard box.bottom() call below
            else:
                bot_mid_ch = "\u253c"  # ┼
                bot_left_ch = "\u251c"  # ├
                bot_right_ch = "\u2524"  # ┤
                bot_line = _build_row_border(bot_left_ch, box._bottombar, bot_mid_ch, bot_right_ch)
                box._emit(bot_line, "bottom")

        _key_vars = getattr(p, 'key_vars', []) or []
        _var_sens_data = {}
        for _vpname in _names:
            _vars_key = f"sensitivity_of_vars_to_{_vpname}"
            if _vars_key in sd and _key_vars:
                _var_sens_data[_vpname] = sd[_vars_key]

        if _key_vars and _var_sens_data:
            _var_col_data = []
            _var_col_w_needed = 13

            for pi, pname in enumerate(_names):
                pvals = _pvals[pi]
                _vdata = _var_sens_data.get(pname, [])
                if not _vdata:
                    continue
                _bi_v = 0
                if _control_indices and pname in _control_indices:
                    _bi_v = _control_indices[pname]
                elif _default_base_idx < len(pvals):
                    _bi_v = _default_base_idx

                def _sort_key_var(idx):
                    pv = pvals[idx]
                    if isinstance(pv, (_np.ndarray, list, tuple)):
                        _arr = _np.asarray(pv, dtype=float)
                        if _arr.ndim == 2 and _arr.shape[1] == 2:
                            _centroid = _np.mean(_arr, axis=0)
                            _spread = float(_np.mean(_np.abs(_arr - _centroid)))
                            return (-_arr.shape[0], -_spread)
                        return float(_np.mean(_arr))
                    try:
                        return float(pv)
                    except (TypeError, ValueError):
                        return 0.0

                _base_val = _sort_key_var(_bi_v)
                _lower = sorted([i for i in range(len(pvals)) if i != _bi_v and _sort_key_var(i) < _base_val], key=_sort_key_var)
                _upper = sorted([i for i in range(len(pvals)) if i != _bi_v and _sort_key_var(i) >= _base_val], key=_sort_key_var)
                _sorted_indices = _lower + [_bi_v] + _upper

                _col_lines = []
                _hdr_parts = []
                for _vk in _key_vars:
                    _hdr_parts.append(f"{'Val':>6} {_vk:>6}")
                _col_lines.append(" " + " ".join(_hdr_parts))

                for i in _sorted_indices:
                    if i >= len(_vdata) or not isinstance(_vdata[i], dict):
                        continue
                    _pval = pvals[i]
                    _pval_pct = _pval_pct_str(_pval, pvals[_bi_v])

                    _var_cells = []
                    for _vk in _key_vars:
                        _val = _vdata[i].get(_vk)
                        if _val is None:
                            _var_cells.append(f"{'-':>6}")
                            continue
                        try:
                            _var_v = float(_np.mean(_np.asarray(_val, dtype=float)))
                        except Exception:
                            _var_cells.append(f"{'-':>6}")
                            continue

                        _base_vv = None
                        if _bi_v < len(_vdata) and isinstance(_vdata[_bi_v], dict):
                            _bval = _vdata[_bi_v].get(_vk)
                            if _bval is not None:
                                try:
                                    _base_vv = float(_np.mean(_np.asarray(_bval, dtype=float)))
                                except Exception:
                                    pass

                        if _base_vv is not None and abs(_base_vv) > 1e-10:
                            _var_pct = (_var_v - _base_vv) / abs(_base_vv) * 100
                            _var_cells.append(f"{_var_pct:+.1f}")
                        else:
                            _var_cells.append(f"{'-':>6}")

                    _line = f" {_pval_pct:>6} " + " ".join(f"{c:>6}" for c in _var_cells)
                    _lv = box._visible_len(_line)
                    if _lv > _var_col_w_needed:
                        _var_col_w_needed = _lv
                    _col_lines.append(_line)

                _var_col_data.append({'name': pname, 'lines': _col_lines})

            # Decision variable sensitivity is now integrated into main sensitivity boxes
            # The following section for separate variable boxes is disabled
            if False:  # Disabled - integrated into sensitivity boxes above
                _var_col_w_min = max(13, _var_col_w_needed)
                _var_n_cols = min(len(_var_col_data), 5)
                while _var_n_cols > 1:
                    _tw = _var_n_cols * _var_col_w_min + (_var_n_cols - 1) * sep_w
                    if _tw <= inner_width:
                        break
                    _var_n_cols -= 1
                _var_rows_of_cols = []
                for start in range(0, len(_var_col_data), _var_n_cols):
                    _var_rows_of_cols.append(_var_col_data[start:start + _var_n_cols])

                _var_col_w = max(13, _var_col_w_needed)
                _var_total_needed = _var_n_cols * _var_col_w + (_var_n_cols - 1) * sep_w
                if _var_total_needed > inner_width and _var_n_cols > 0:
                    _var_col_w = max(10, (inner_width - (_var_n_cols - 1) * sep_w) // _var_n_cols)

                for _vr_idx, _vr_data in enumerate(_var_rows_of_cols):
                    _vr_actual_n = len(_vr_data)
                    _vr_sep_positions = []
                    for ci in range(_vr_actual_n - 1):
                        pos = 1 + 1 + (ci + 1) * _var_col_w + ci * sep_w + 1
                        _vr_sep_positions.append(pos)

                    def _build_var_border(left_ch, bar_ch, mid_ch, right_ch, label=""):
                        line = [left_ch] + [bar_ch] * (box.width - 2) + [right_ch]
                        for sp in _vr_sep_positions:
                            if 0 < sp < len(line):
                                line[sp] = mid_ch
                        if label:
                            label_str = f" {label} "
                            for i, ch in enumerate(label_str):
                                if 1 + i < len(line) - 1:
                                    line[1 + i] = ch
                        return "".join(line)

                    _var_lbl = "" if _vr_idx == 0 else ""
                    _var_top = _build_var_border(box._topleft, box._topbar, box._middletop, box._topright, label=_var_lbl)
                    box._emit(_var_top, "top")

                    _vr_hdr_cells = []
                    for _vr_ci, _vr_cd in enumerate(_vr_data):
                        _vr_cw = _var_col_w
                        _vr_hdr = f" {_vr_cd['name']}"
                        _vr_hvw = box._visible_len(_vr_hdr)
                        if _vr_hvw < _vr_cw:
                            _vr_hdr = _vr_hdr.ljust(_vr_cw)
                        elif _vr_hvw > _vr_cw:
                            _vr_hdr = _vr_hdr[:_vr_cw]
                        _vr_hdr_cells.append(_vr_hdr)
                    _vr_hdr_inner = sep.join(_vr_hdr_cells)
                    _vr_pad = inner_width - box._visible_len(_vr_hdr_inner)
                    if _vr_pad > 0:
                        _vr_hdr_inner += " " * _vr_pad
                    box._emit(f"{box._border} {_vr_hdr_inner} {box._border}", "row")

                    _vr_max_h = max(len(d['lines']) for d in _vr_data) if _vr_data else 0
                    for _vr_d in _vr_data:
                        while len(_vr_d['lines']) < _vr_max_h:
                            _vr_d['lines'].append(" " * _var_col_w)

                    for _vr_ri in range(_vr_max_h):
                        _vr_cells = []
                        for _vr_ci, _vr_cd in enumerate(_vr_data):
                            _vr_line = _vr_cd['lines'][_vr_ri]
                            _vr_lv = box._visible_len(_vr_line)
                            if _vr_lv < _var_col_w:
                                _vr_line = _vr_line + " " * (_var_col_w - _vr_lv)
                            elif _vr_lv > _var_col_w:
                                _vr_line = _vr_line[:_var_col_w]
                            _vr_cells.append(_vr_line)
                        _vr_row_inner = sep.join(_vr_cells)
                        _vr_rpad = inner_width - box._visible_len(_vr_row_inner)
                        if _vr_rpad > 0:
                            _vr_row_inner += " " * _vr_rpad
                        box._emit(f"{box._border} {_vr_row_inner} {box._border}", "row")

                    _vr_is_last = (_vr_idx == len(_var_rows_of_cols) - 1)
                    _vr_bot_mid = "\u2534" if _vr_is_last else "\u253c"
                    _vr_bot = _build_var_border("\u251c", box._bottombar, _vr_bot_mid, "\u2524")
                    box._emit(_vr_bot, "bottom")

        # Interaction data is now integrated into sensitivity boxes above
        # The separate interaction section is disabled
        if False and _interaction_data:
            """
            Pairwise Interaction Analysis
            
            Evaluates how parameters interact when both change simultaneously.
            Reports three essential metrics, all normalized [0,1]:
            
            1. Str (Strength): Interaction effect relative to main effects
               - Formula: |I_ab| / (|Z10-Z00| + |Z01-Z00|)
               - 0 = no interaction (perfectly additive)
               - 1 = interaction dominates main effects
               - Interpretation: >0.15 considered significant
               
            2. Non (Nonlinearity): Deviation from linear additivity
               - Formula: |I_ab| / |Z11-Z00|
               - 0 = perfectly linear/additive system
               - 1 = highly nonlinear behavior
               - Measures curvature in parameter-objective relationship
               
            3. Dec (Decision Change): Solution stability under joint variation
               - Fraction of decision variables that changed (0 to 1)
               - 0 = decisions stable (parametric change only)
               - 1 = all decisions changed (structural change)
               - Indicates robustness of solution structure
            
            Type labels:
            - Synergy: Parameters amplify each other (I_ab > 0)
            - Antagonism: Parameters counteract each other (I_ab < 0)
            
            Theoretical basis: Two-factor interaction analysis from factorial 
            design (DOE) principles. Interaction term: I_ab = Z11 - Z10 - Z01 + Z00
            """
            
            # Build interaction data
            _int_rows = []
            for ix in _interaction_data:
                _pair = ix['pair']
                
                _r_ab = 0.0
                if ix['i_ab'] is not None and ix['z00'] is not None and ix['z10'] is not None and ix['z01'] is not None:
                    _main_a = abs(ix['z10'] - ix['z00'])
                    _main_b = abs(ix['z01'] - ix['z00'])
                    _denom = _main_a + _main_b
                    _r_ab = abs(ix['i_ab']) / _denom if _denom > 0 else 0.0
                
                _i_type = "None"
                if ix['i_ab'] is not None:
                    if ix['i_ab'] > 0:
                        _i_type = "Synergy"
                    elif ix['i_ab'] < 0:
                        _i_type = "Antagonism"
                
                _int_rows.append({
                    'pair': _pair,
                    'type': _i_type,
                    'term': ix['i_ab'],
                    'strength': _r_ab,
                    'z00': ix['z00'],
                    'z10': ix['z10'],
                    'z01': ix['z01'],
                    'z11': ix['z11'],
                    'd_10': ix['d_10'],
                    'd_01': ix['d_01'],
                    'd_11': ix['d_11']
                })
            
            # Display as multi-column boxed structure (3 per row like sensitivity)
            _n_int = len(_int_rows)
            if _n_int > 0:
                _int_n_cols = 3  # Always 3 columns for interactions
                _int_col_width = (box.width - 2) // _int_n_cols
                
                # Build column data for each interaction - only essential metrics [0,1]
                _int_col_data = []
                for _ir in _int_rows:
                    # Interaction strength (already 0-1, ratio of interaction to main effects)
                    _strength = _ir['strength']
                    
                    # Decision change fraction (0=stable, 1=fully changed)
                    _dec_chg = _ir['d_11']  # Both parameters changed
                    
                    # Nonlinearity indicator: how much interaction deviates from additivity
                    # Normalized by comparing interaction term to total effect magnitude
                    if _ir['z00'] is not None and _ir['z11'] is not None:
                        _total_effect = abs(_ir['z11'] - _ir['z00'])
                        if _total_effect > 1e-10:
                            _nonlin = min(abs(_ir['term']) / _total_effect, 1.0) if _ir['term'] is not None else 0
                        else:
                            _nonlin = 0
                    else:
                        _nonlin = 0
                    
                    # Display format: 
                    # Row 1: Pair name and effect type
                    # Row 2: Column headers
                    # Row 3: Values
                    _lines = [
                        f"  {_ir['pair']:<14} {_ir['type']:>8}",
                        f"   Str    Non    Dec",
                        f"  {_strength:>4.2f}   {_nonlin:>4.2f}   {_dec_chg:>4.2f}",
                    ]
                    _int_col_data.append({
                        'label': _ir['pair'],
                        'lines': _lines,
                        'effect': _ir['type'].lower()
                    })
                
                # Emit in rows of 3 columns - respect box width
                _inner_width = box.width - 2
                for _start_idx in range(0, _n_int, _int_n_cols):
                    _cols_in_row = _int_col_data[_start_idx:_start_idx + _int_n_cols]
                    _n_in_row = len(_cols_in_row)
                    
                    # Calculate widths to fit within box.width, accounting for separators
                    _sep_count = _n_in_row - 1
                    _avail_for_cols = _inner_width - _sep_count
                    
                    if _n_in_row == 1:
                        _widths = [_avail_for_cols]
                    elif _n_in_row == 2:
                        _w = _avail_for_cols // 2
                        _widths = [_w, _avail_for_cols - _w]
                    else:  # 3 columns
                        _w = _avail_for_cols // 3
                        _widths = [_w, _w, _avail_for_cols - 2*_w]
                    
                    # Top border - simple bars only (pair names inside boxes)
                    _parts = []
                    for _i, _col in enumerate(_cols_in_row):
                        _w = _widths[_i]
                        _top_str = box._topbar * _w
                        _parts.append(_top_str)
                    
                    _sep_char = box._middletop
                    _row_str = box._topleft + _sep_char.join(_parts) + box._topright
                    box._emit(_row_str, "top")
                    
                    # Data rows
                    _max_lines = max(len(_col['lines']) for _col in _cols_in_row)
                    for _line_idx in range(_max_lines):
                        _parts = []
                        for _i, _col in enumerate(_cols_in_row):
                            _w = _widths[_i]
                            _inner_w = _w - 2
                            _txt = _col['lines'][_line_idx] if _line_idx < len(_col['lines']) else ""
                            _visible = box._visible_len(_txt)
                            _pad = _inner_w - _visible
                            if _pad > 0:
                                _txt += " " * _pad
                            elif _pad < 0:
                                _txt = _txt[:_inner_w]
                            _parts.append(f" {_txt} ")
                        _row_str = box._border + box._border.join(_parts) + box._border
                        box._emit(_row_str, "row")
                    
                    # Bottom border with separators (only for last row of interactions)
                    if _start_idx + _int_n_cols >= _n_int:
                        _bot_parts = []
                        for _i in range(_n_in_row):
                            _w = _widths[_i]
                            _bot_parts.append(box._bottombar * _w)
                        _bot_sep = box._middlebottom  # ┴ character
                        _bot_str = box._bottomleft + _bot_sep.join(_bot_parts) + box._bottomright
                        
                        # Add elapsed time to the bottom border
                        elapsed = p.sensitivity_end_timer - p.sensitivity_begin_timer
                        _time_str = format_time_duration(elapsed, name="")
                        if _time_str:
                            # Insert time at the right end
                            _time_with_space = f" {_time_str} "
                            _time_len = len(_time_with_space)
                            if len(_bot_str) >= _time_len + 2:
                                _bot_str = _bot_str[:-_time_len-1] + _time_with_space + box._bottomright
                        
                        box._emit(_bot_str, "bottom")
                        return  # Skip the standard box.bottom() call below

        # Fallback: no parameter (or variance) panel rendered, so no top
        # border was emitted and there is nothing to close.  (The
        # constraint-batch impact moved to the Diagnostics box.)
        if not _rows_of_cols:
            if not box._buffered:
                box.render()
            return

        elapsed = p.sensitivity_end_timer - p.sensitivity_begin_timer
        box.bottom(right=format_time_duration(elapsed, name=""))

        if not box._buffered:
            box.render()

    def compute_impact(self, pname, pvals, bi, sd):
        """Aggregate sensitivity-impact metrics for one swept parameter.

        Returns the entry the Sensitivity box renders as its footer --
        ``sensitivity`` (Sen), ``decision`` (Dec), ``asymmetry`` (Asy),
        ``uncertainty`` (Unc), ``cpt_sensitivity`` (Tim) and
        ``weighted_impact`` (Imp) -- plus ``direction_effect``, the glyph
        behind the parameter marker (raise / lower / none / mixed).

        Both the report and :meth:`get_impact` run this one implementation,
        so the printed numbers and the queried ones can never disagree.

        Parameters
        ----------
        pname : str
            Name of the swept parameter.
        pvals : sequence
            Scenario values of ``pname``.
        bi : int
            Index of the control (baseline) scenario.
        sd : dict
            ``p.sensitivity_data``.
        """
        p = self._p
        import numpy as _np
        _is_single = p.number_of_objectives == 1
        _directions = getattr(p, 'directions', []) or []
        _obj_dir = _directions[0] if _directions else 'min'
        _obj_weights = getattr(p, 'sensitivity_obj_weights', None)

        def _aggregate_objective(obj_val):
            if obj_val is None:
                return None
            if _is_single:
                try:
                    return float(obj_val)
                except (TypeError, ValueError):
                    return None
            arr = _np.asarray(obj_val, dtype=float)
            if arr.ndim == 0:
                return float(arr)
            if arr.size == 0:
                return None
            if _obj_weights is not None:
                w = _np.asarray(_obj_weights, dtype=float)
                if w.shape == arr.shape:
                    return float(_np.sum(arr * w) / _np.sum(w))
            return float(_np.mean(arr))

        """
        Compute sensitivity impact metrics for a parameter.

        Metrics (displayed as Sen, Dec, Unc, Tim, Asy, Imp):

        1. sensitivity (Sen): Objective elasticity - median of |Δobj%| / |Δparam%|
           - Measures how responsive the objective is to parameter changes
           - Value of 0.5 means 1% parameter change causes ~0.5% objective change
           - Theoretically: this is the classical price elasticity measure
           - Robust: uses median to handle outliers, separates positive/negative directions

        2. decision (Dec): Decision structure impact - fraction of variables that change
           - Measures what proportion of decision variables are affected
           - Value of 1.0 means all decision variables change with parameter
           - Theoretically: indicates structural vs. marginal parameter importance
           - Robust: counts actual solution differences, filters auxiliary variables

        3. uncertainty (Unc): Output variability - CV of objectives across scenarios
           - Measures coefficient of variation: std(objectives) / mean(|objectives|)
           - Value of 0.2 means objective varies by ~20% across parameter scenarios
           - Theoretically: quantifies output risk/uncertainty from parameter variation
           - Robust: uses absolute mean to avoid sign issues in objectives

        4. cpt_sensitivity (Tim): Computation time elasticity - median of |Δcpt%| / |Δparam%|
           - Measures how computation time scales with parameter changes
           - Value of 1.35 means 1% parameter change causes 1.35% time change
           - Theoretically: identifies computational bottlenecks
           - Robust: same elasticity method as sensitivity

        5. asymmetry (Asy): Directional asymmetry index
           - Formula: |s_plus - s_minus| / (s_plus + s_minus)
           - Value of 0 means symmetric response, 1 means completely one-sided
           - Theoretically: detects non-linear or threshold behavior
           - Robust: normalized ratio, bounded [0,1]

        6. weighted_impact (Imp): Combined importance - sensitivity × decision
           - Combines magnitude of change with structural importance
           - High values indicate parameters that both move objective and restructure solution
           - Theoretically: prioritizes parameters with broad impact
           - Robust: product of two independently valid metrics

        All formulations follow established sensitivity analysis theory and are
        defensible for optimization under uncertainty.
        """
        _impact = {'name': pname, 'sensitivity': 0.0, 'cpt_sensitivity': 0.0,
                   'decision': 0.0, 'asymmetry': 0.0, 'uncertainty': 0.0,
                   'weighted_impact': 0.0, 'direction_effect': '─'}
        obj_key = f"sensitivity_of_objectives_to_{pname}"
        sol_key = f"sensitivity_of_solutions_to_{pname}"
        cpt_key = f"sensitivity_of_cpt_to_{pname}"
        obj_data = sd.get(obj_key, [])
        sol_data = sd.get(sol_key, [])
        cpt_data = sd.get(cpt_key, [])
        if not obj_data or len(obj_data) < 2:
            return _impact
        _base_obj_raw = None
        try:
            if _is_single:
                _base_obj_raw = float(p.objective_values[0][0]) if p.objective_values is not None else None
            else:
                _base_obj_raw = p.objective_values
        except Exception:
            pass
        _base_obj = _aggregate_objective(_base_obj_raw)
        _valid_objs = [o for o in obj_data if o is not None]
        if not _valid_objs or _base_obj is None:
            return _impact
        _nums = []
        for o in _valid_objs:
            agg = _aggregate_objective(o)
            if agg is not None:
                _nums.append(agg)
        if len(_nums) < 2:
            return _impact
        # Build a mapping from scenario index to valid nums index
        # (skip None/infeasible scenarios)
        _idx_map = {}
        _valid_counter = 0
        for _oi, _ov in enumerate(obj_data):
            if _ov is not None:
                agg = _aggregate_objective(_ov)
                if agg is not None:
                    _idx_map[_oi] = _valid_counter
                    _valid_counter += 1
        _base_idx = bi
        if _base_idx >= len(pvals) or _base_idx not in _idx_map:
            # Find the base in valid indices
            for _try_idx in range(len(pvals)):
                if _try_idx in _idx_map:
                    _base_idx = _try_idx
                    break
        _base_o = _nums[_idx_map[_base_idx]] if _base_idx in _idx_map else _base_obj

        # ============================================================================
        # SENSITIVITY & TIME SENSITIVITY: Elasticity-based measures
        # Formula: median(|Δoutput%| / |Δparam%|) for positive and negative changes
        # Theoretical basis: Classical elasticity from economics, robust to outliers
        # ============================================================================
        _s_plus = 0.0
        _s_minus = 0.0
        _obj_changes_pos = []
        _obj_changes_neg = []
        _cpt_changes_pos = []
        _cpt_changes_neg = []
        _base_cpt = cpt_data[_base_idx] if _base_idx < len(cpt_data) else None
        _base_pval = pvals[_base_idx]
        _base_pval_num = float(_base_pval) if isinstance(_base_pval, (int, float)) else float(_np.mean(_base_pval))

        for _ci2, _cv2 in enumerate(pvals):
            if _ci2 == _base_idx:
                continue
            if _ci2 not in _idx_map:
                continue  # Skip infeasible scenarios
            try:
                _cval2 = float(_cv2) if isinstance(_cv2, (int, float)) else float(_np.mean(_cv2))
                _param_pct_chg = (_cval2 - _base_pval_num) / abs(_base_pval_num) if abs(_base_pval_num) > 1e-10 else 0
                _obj_pct_chg = (_nums[_idx_map[_ci2]] - _base_o) / abs(_base_o) if abs(_base_o) > 1e-10 else 0

                # Elasticity: (% change in objective) / (% change in parameter)
                if abs(_param_pct_chg) > 1e-10:
                    _elasticity = abs(_obj_pct_chg) / abs(_param_pct_chg)
                    if _cval2 > _base_pval_num:
                        _obj_changes_pos.append(_elasticity)
                    elif _cval2 < _base_pval_num:
                        _obj_changes_neg.append(_elasticity)

                # Same for CPT
                if cpt_data and _ci2 < len(cpt_data) and cpt_data[_ci2] is not None and _base_cpt is not None and abs(_base_cpt) > 1e-10:
                    _cpt_pct_chg = (cpt_data[_ci2] - _base_cpt) / abs(_base_cpt)
                    if abs(_param_pct_chg) > 1e-10:
                        _cpt_elasticity = abs(_cpt_pct_chg) / abs(_param_pct_chg)
                        if _cval2 > _base_pval_num:
                            _cpt_changes_pos.append(_cpt_elasticity)
                        elif _cval2 < _base_pval_num:
                            _cpt_changes_neg.append(_cpt_elasticity)
            except Exception:
                pass

        # Use median elasticity as sensitivity measure (robust to outliers)
        _s_plus = float(_np.median(_obj_changes_pos)) if _obj_changes_pos else 0.0
        _s_minus = float(_np.median(_obj_changes_neg)) if _obj_changes_neg else 0.0
        _impact['sensitivity'] = (_s_plus + _s_minus) / 2.0

        # CPT sensitivity (computation time elasticity)
        if _cpt_changes_pos or _cpt_changes_neg:
            _cpt_s_plus = float(_np.median(_cpt_changes_pos)) if _cpt_changes_pos else 0.0
            _cpt_s_minus = float(_np.median(_cpt_changes_neg)) if _cpt_changes_neg else 0.0
            _impact['cpt_sensitivity'] = (_cpt_s_plus + _cpt_s_minus) / 2.0

        # ============================================================================
        # ASYMMETRY: Directional response asymmetry
        # Formula: |s_plus - s_minus| / (s_plus + s_minus), bounded [0,1]
        # 0 = symmetric response, 1 = completely one-sided
        # ============================================================================
        if (_s_plus + _s_minus) > 0:
            _impact['asymmetry'] = abs(_s_plus - _s_minus) / (_s_plus + _s_minus)
        # ============================================================================
        # DECISION IMPACT: Fraction of decision variables that change
        # Counts how many solution variables differ from baseline
        # Filters out auxiliary variables (autolin, sos2, rdiv, internal)
        # ============================================================================
        _d_j = 0.0
        _base_sol = sol_data[_base_idx] if _base_idx < len(sol_data) else None
        if isinstance(_base_sol, dict) and sol_data:
            _flat_base = _base_sol
            if not _is_single:
                for _pk, _pv in _base_sol.items():
                    if isinstance(_pv, dict):
                        _flat_base = _pv
                        break
            _n_total = 0
            _n_changed = 0
            for _si, _sv in enumerate(sol_data):
                if _si == _base_idx or not isinstance(_sv, dict):
                    continue
                _flat_sv = _sv
                if not _is_single:
                    for _pk, _pv in _sv.items():
                        if isinstance(_pv, dict):
                            _flat_sv = _pv
                            break
                for k, v in _flat_base.items():
                    if not isinstance(k, str):
                        continue
                    if k.startswith('_') or 'autolin' in k or 'sos2' in k or 'rdiv' in k:
                        continue
                    if k not in _flat_sv:
                        continue
                    _n_total += 1
                    _vv = _flat_sv[k]
                    if isinstance(v, _np.ndarray) and isinstance(_vv, _np.ndarray):
                        if not _np.array_equal(v, _vv):
                            _n_changed += 1
                    elif v != _vv:
                        _n_changed += 1
            if _n_total > 0:
                _d_j = _n_changed / _n_total
        _impact['decision'] = _d_j
        _all_changes = []
        for _ci2, _cv2 in enumerate(pvals):
            if _ci2 == _base_idx:
                continue
            if _ci2 not in _idx_map:
                continue  # Skip infeasible scenarios
            try:
                _cval2 = float(_cv2) if isinstance(_cv2, (int, float)) else float(_np.mean(_cv2))
                _bval2 = float(pvals[_base_idx]) if isinstance(pvals[_base_idx], (int, float)) else float(_np.mean(pvals[_base_idx]))
                _obj_chg = (_nums[_idx_map[_ci2]] - _base_o) / abs(_base_o) if _base_o != 0 else 0
                _param_chg = _cval2 - _bval2
                if _param_chg != 0:
                    _all_changes.append((_param_chg, _obj_chg))
            except Exception:
                pass
        if _all_changes:
            _pos_param_changes = [c for pc, c in _all_changes if pc > 0]
            _neg_param_changes = [c for pc, c in _all_changes if pc < 0]
            _avg_pos_obj = float(_np.mean(_pos_param_changes)) if _pos_param_changes else 0
            _avg_neg_obj = float(_np.mean(_neg_param_changes)) if _neg_param_changes else 0
            # Action direction: which way to move THIS parameter to improve
            # the objective (independent of max/min sense):
            #   ↑ raise it    ↓ lower it    ─ no effect    ↕ mixed
            # A bucket is "significant" when its mean relative objective
            # change clears the 0.1% threshold.
            if _obj_dir == 'max':
                _pos_helps = _avg_pos_obj > 0.001    # raising it raised the objective
                _neg_helps = _avg_neg_obj > 0.001    # lowering it raised the objective
            else:
                _pos_helps = _avg_pos_obj < -0.001   # raising it lowered the objective
                _neg_helps = _avg_neg_obj < -0.001   # lowering it lowered the objective
            _pos_sig = abs(_avg_pos_obj) > 0.001
            _neg_sig = abs(_avg_neg_obj) > 0.001
            _act_pos = "↑" if _pos_helps else "↓"    # advice from raising scenarios
            _act_neg = "↓" if _neg_helps else "↑"    # advice from lowering scenarios
            if _pos_sig and _neg_sig:
                # Both directions observed: agree -> one direction, disagree
                # (non-monotone, optimum at/near the base) -> flag it.
                _impact['direction_effect'] = _act_pos if _act_pos == _act_neg else "↕"
            elif _pos_sig:
                _impact['direction_effect'] = _act_pos
            elif _neg_sig:
                _impact['direction_effect'] = _act_neg
            else:
                _impact['direction_effect'] = "─"

        # ============================================================================
        # UNCERTAINTY: Output variability across scenarios
        # Formula: CV = std(objectives) / mean(|objectives|)
        # Measures risk/variability in outcomes, not input parameter variation
        # ============================================================================
        if len(_nums) > 1:
            _obj_std = float(_np.std(_nums))
            _obj_mean = float(_np.mean([abs(x) for x in _nums]))
            if _obj_mean > 1e-10:  # Avoid division by very small numbers
                _impact['uncertainty'] = _obj_std / _obj_mean
            else:
                _impact['uncertainty'] = 0.0
        else:
            _impact['uncertainty'] = 0.0

        # ============================================================================
        # WEIGHTED IMPACT: Combined importance metric
        # Formula: sensitivity × decision
        # Prioritizes parameters with both high elasticity and structural impact
        # ============================================================================
        _impact['weighted_impact'] = _impact['sensitivity'] * _impact['decision']

        return _impact

    def get_impact(self, parameter_name):
        """Impact of one swept parameter, as data (nothing is printed).

        Returns

        -------
        dict
            ``parameter`` -- the name as given.

            ``control_index`` -- which scenario is the baseline every
            percentage is measured against.

            ``values`` -- the scenario values, in recorded order.

            ``action`` -- the recommended move, one of ``"raise"``,
            ``"lower"``, ``"none"`` or ``"mixed"``; the word shown beside
            the marker in the parameter box.

            ``metrics`` -- the aggregate entry of :meth:`compute_impact`
            (Sen/Dec/Asy/Unc/Tim/Imp plus the ``direction_effect`` glyph).

            ``scenarios`` -- one dict per scenario, in recorded order:

            ``index``, ``value``, ``control``
                position and value of the scenario.
            ``health``
                ``True`` healthy, ``False`` infeasible, ``None`` when the
                build or run raised.
            ``objective``
                objective value(s) exactly as recorded; a multi-objective
                run keeps one row per Pareto point.
            ``objective_means``
                one entry per objective -- the front average the report's
                ``Obj1..ObjM`` columns are built on (a one-element list for
                a single-objective run).
            ``objective_pct``
                percent change of each entry of ``objective_means`` against
                the control scenario, ``None`` where the control is 0.
            ``cpt``, ``cpt_pct``
                solve time and its percent change against the control.
            ``solution``
                recorded solution, carrying the ``m.term()`` values.

        Raises
        ------
        ValueError
            When ``parameter_name`` was never swept.
        """
        p = self._p
        _names = list(getattr(p, 'sensitivity_parameter_names', None) or [])
        if parameter_name not in _names:
            raise ValueError(
                "no sensitivity sweep for parameter %r; swept parameters: %s"
                % (parameter_name, ", ".join(_names) if _names else "(none)"))
        sd = getattr(p, 'sensitivity_data', None) or {}
        _pvals_all = list(
            getattr(p, 'sensitivity_parameter_values', None) or [])
        _pi = _names.index(parameter_name)
        pvals = _pvals_all[_pi] if _pi < len(_pvals_all) else []
        _ctrl = getattr(p, 'sensitivity_control_indices', None)
        if isinstance(_ctrl, dict) and isinstance(_ctrl.get(parameter_name), int):
            bi = _ctrl[parameter_name]
        else:
            _d = getattr(p, 'control_scenario', 0)
            bi = _d if _d < len(pvals) else 0

        obj_data = list(sd.get(f"sensitivity_of_objectives_to_{parameter_name}") or [])
        sol_data = list(sd.get(f"sensitivity_of_solutions_to_{parameter_name}") or [])
        cpt_data = list(sd.get(f"sensitivity_of_cpt_to_{parameter_name}") or [])
        health_data = list(sd.get(f"sensitivity_of_health_to_{parameter_name}") or [])
        n = max(len(obj_data), len(sol_data), len(cpt_data), len(health_data),
                len(pvals))

        _n_obj = p.number_of_objectives
        _base_obj = obj_data[bi] if bi < len(obj_data) else None
        _base_cpt = cpt_data[bi] if bi < len(cpt_data) else None

        def _pct(cur, base):
            try:
                _b = float(base)
                _c = float(cur)
            except (TypeError, ValueError):
                return None
            if abs(_b) <= 1e-10:
                return None
            return (_c - _b) / abs(_b) * 100.0

        _bmeans = (ReportEngine._objective_means(_base_obj, _n_obj)
                   if _base_obj is not None else None)
        scenarios = []
        for _i in range(n):
            _ov = obj_data[_i] if _i < len(obj_data) else None
            _means = (ReportEngine._objective_means(_ov, _n_obj)
                      if _ov is not None else None)
            _o_pct = None
            if _means is not None and _bmeans is not None:
                _o_pct = [_pct(_c, _b)
                          for _c, _b in zip(_means, _bmeans)]
            _cv = cpt_data[_i] if _i < len(cpt_data) else None
            scenarios.append({
                'index': _i,
                'value': pvals[_i] if _i < len(pvals) else None,
                'control': _i == bi,
                'health': health_data[_i] if _i < len(health_data) else None,
                'objective': _ov,
                'objective_means': _means,
                'objective_pct': _o_pct,
                'cpt': _cv,
                'cpt_pct': _pct(_cv, _base_cpt),
                'solution': sol_data[_i] if _i < len(sol_data) else None,
            })

        try:
            _metrics = self.compute_impact(parameter_name, pvals, bi, sd)
        except Exception:
            _metrics = {'name': parameter_name, 'sensitivity': 0.0,
                        'cpt_sensitivity': 0.0, 'decision': 0.0,
                        'asymmetry': 0.0, 'uncertainty': 0.0,
                        'weighted_impact': 0.0, 'direction_effect': '─'}
        _glyph = _metrics.get('direction_effect', '─')
        return {'parameter': parameter_name,
                'control_index': bi,
                'values': list(pvals),
                'action': DIRECTION_HINTS.get(_glyph, _glyph),
                'metrics': _metrics,
                'scenarios': scenarios}

    @staticmethod
    def _problem_batches(registry):
        """Constraint batches that belong to the problem the user wrote.

        FelooPy injects rows of its own while solving -- NWSM's
        ``_weighted_k`` linking rows, payoff rows, Benders/DW/CG cuts -- and
        registers them as batches under an '_'-prefixed name.  They are
        implementation details rather than constraints of the problem, so
        they are neither impact-diagnosed nor removed while isolating an
        infeasibility.  Linearization adds its constraints unnamed and
        unnamed constraints are never batches; ``_collect_lp_analysis``
        already applies the same '_' rule to constraint labels.
        """
        return {k: v for k, v in (registry or {}).items()
                if isinstance(k, str) and not k.startswith('_')}

    @staticmethod
    def _objective_means(obj_val, n_obj=None):
        """Average value of each objective across the Pareto front.

        A multi-objective run has no single objective to quote, so every
        consumer -- Sensitivity rows, batch-impact rows, interaction
        z-values -- reads this per-objective average instead of a raw
        objective vector.  Returns one entry per objective, or ``None``
        when there is nothing to average.
        """
        import numpy as _np
        if obj_val is None:
            return None
        try:
            arr = _np.asarray(obj_val, dtype=float)
        except (TypeError, ValueError):
            return None
        if arr.size == 0:
            return None
        if arr.ndim == 0:
            return [float(arr)]
        if arr.ndim == 1:
            if n_obj is not None and n_obj > 1 and arr.size == n_obj:
                # a single Pareto point already holds one value per objective
                return [float(v) for v in arr]
            return [float(_np.mean(arr))]
        return [float(_np.mean(arr[:, k])) for k in range(arr.shape[1])]

    @staticmethod
    def _aggregate_objective_static(obj_val, is_single, obj_weights=None):
        import numpy as _np
        if obj_val is None:
            return None
        if is_single:
            try:
                return float(obj_val)
            except (TypeError, ValueError):
                return None
        arr = _np.asarray(obj_val, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        if arr.size == 0:
            return None
        if obj_weights is not None:
            w = _np.asarray(obj_weights, dtype=float)
            if w.shape == arr.shape:
                return float(_np.sum(arr * w) / _np.sum(w))
        return float(_np.mean(arr))

    @staticmethod
    def _compute_decision_change(sol_0, sol_1):
        import numpy as _np
        if not isinstance(sol_0, dict) or not isinstance(sol_1, dict):
            return 0.0
        _flat_0 = sol_0
        _flat_1 = sol_1
        for _pk, _pv in sol_0.items():
            if isinstance(_pv, dict):
                _flat_0 = _pv
                break
        for _pk, _pv in sol_1.items():
            if isinstance(_pv, dict):
                _flat_1 = _pv
                break
        _n_total = 0
        _n_changed = 0
        for k, v in _flat_0.items():
            if not isinstance(k, str):
                continue
            if k.startswith('_') or 'autolin' in k or 'sos2' in k or 'rdiv' in k:
                continue
            if k not in _flat_1:
                continue
            _n_total += 1
            _vv = _flat_1[k]
            if isinstance(v, _np.ndarray) and isinstance(_vv, _np.ndarray):
                if not _np.array_equal(v, _vv):
                    _n_changed += 1
            elif v != _vv:
                _n_changed += 1
        return _n_changed / _n_total if _n_total > 0 else 0.0

    def _try_auto_sensitivity(self, p):
        import time as _time
        import copy as _copy
        import numpy as _np

        _dataset = getattr(p, 'inputdata', None)
        if _dataset is None or not hasattr(_dataset, 'data'):
            return None

        _existing_key_params = getattr(p, 'key_params', []) or []
        _existing_scenarios = getattr(p, 'scenarios', []) or []

        if _existing_key_params and _existing_scenarios:
            _key_params = _existing_key_params
            _scenarios = _existing_scenarios
        else:
            _data_keys = list(_dataset.data.keys())
            if not _data_keys:
                return None

            _key_params = []
            for k in _data_keys:
                v = _dataset.data[k]
                if isinstance(v, (set, dict, str, bytes)) or v is None:
                    # structural domains / non-numeric values: nothing to perturb
                    continue
                if isinstance(v, (int, float)) or hasattr(v, 'size'):
                    # scalar ints/floats, numpy scalars and arrays (scenario
                    # generation shrinks/expands arrays around their centroid)
                    _key_params.append(k)
                elif isinstance(v, (list, tuple)):
                    _key_params.append(k)

            if not _key_params:
                return None

            _scenarios = p._generate_scenarios(_dataset, _key_params)
            if not _scenarios:
                return None

        _env = getattr(p, 'environment', None)
        if _env is None:
            return None

        _key_vars = getattr(p, 'key_vars', []) or []

        from ..operators.data_handler import data_toolkit as _dt
        from ..operators.data_handler import rewrap_scenario as _rewrap
        _result_dataset = _dt(key=0, measure=False)
        _data_keys_store = [
            "sensitivity_values",
            "sensitivity_of_health_to",
            "sensitivity_of_cpt_to",
            "sensitivity_of_objectives_to",
            "sensitivity_of_solutions_to",
            "sensitivity_of_ogr_to",
        ]
        if _key_vars:
            _data_keys_store.append("sensitivity_of_vars_to")

        # no time budget: every parameter and every interaction pair is
        # analyzed to completion (the spinner keeps ticking throughout)
        _start_total = _time.time()
        p.sensitivity_begin_timer = _start_total
        _original_values = {}
        _control_indices = {}

        _orig_dataset_snapshot = _copy.deepcopy(_dataset.data)

        _orig_objective_values = _copy.deepcopy(getattr(p, 'objective_values', None))
        _orig_solutions = _copy.deepcopy(getattr(p, 'solutions', None))
        _orig_em = getattr(p, 'em', None)
        _orig_cpt = getattr(p, 'cpt', None)
        _orig_mgt = getattr(p, 'mgt', None)
        _orig_data = _copy.deepcopy(getattr(p, 'data', None))

        # Boost/decomposition result objects from the search run must not
        # leak into scenario re-solves: _run_impl copies objective values
        # from an existing _cg/_benders/..._result after a fallback or
        # direct solve, which would record the PREVIOUS run's objective as
        # if it belonged to the scenario.  Snapshot them so get_convergence
        # keeps returning the search history after the report finishes.
        _MISSING = object()
        _boost_attrs = ('_benders_result', '_cg_result', '_lagrangian_result',
                        '_branching_result')
        _orig_boost_attrs = {a: getattr(p, a, _MISSING) for a in _boost_attrs}

        def _clear_stale_results():
            for _a in _boost_attrs:
                try:
                    delattr(p, _a)
                except AttributeError:
                    pass
            p.objective_values = None

        from ..helpers.formatter import update_progress

        for pname, pvalues in zip(_key_params, _scenarios):
            update_progress(f"Analyzing the impact of {pname}")
            for key in _data_keys_store:
                _result_dataset.store(f"{key}_{pname}", [])

            _original_value = _copy.deepcopy(_dataset.data[pname])
            _original_values[pname] = _original_value

            _ctrl_idx = 0
            _orig_arr = None
            if isinstance(_original_value, (list, tuple, _np.ndarray)):
                try:
                    _orig_arr = _np.asarray(_original_value, dtype=float)
                except Exception:
                    pass

            for _ci, _cv in enumerate(pvalues):
                try:
                    _is_match = False
                    if _orig_arr is not None:
                        _cv_arr = _np.asarray(_cv, dtype=float)
                        if _cv_arr.shape == _orig_arr.shape and _np.allclose(_cv_arr, _orig_arr, atol=1e-10):
                            _is_match = True
                    elif _cv == _original_value:
                        _is_match = True
                    elif hasattr(_cv, '__float__') and hasattr(_original_value, '__float__'):
                        if abs(float(_cv) - float(_original_value)) < 1e-10:
                            _is_match = True
                    if _is_match:
                        _ctrl_idx = _ci
                        break
                except Exception:
                    pass
            _control_indices[pname] = _ctrl_idx

            for pvalue in pvalues:
                _dataset.data[pname] = _rewrap(_original_value, pvalue)
                _result_dataset.data[f"sensitivity_values_{pname}"].append(pvalue)

                from ..helpers.formatter import suppress_output
                _scenario_failed = False
                try:
                    with suppress_output():
                        _clear_stale_results()
                        p.create_env(_env, verbose=True)
                        p.run(verbose=True)
                except Exception:
                    _scenario_failed = True
                # tri-state health: None = build/run raised (FAIL in the
                # report), False = solver reports infeasible/unhealthy
                # status (INFEAS), True = healthy solve.  Objectives and
                # solutions are recorded only for healthy solves so stale
                # values from a previous run can never leak into the
                # report or the interaction math.
                if _scenario_failed:
                    _health = None
                else:
                    try:
                        _health = p.em.healthy()
                    except Exception:
                        _health = None
                _usable = _health is True
                _result_dataset.data[f"sensitivity_of_health_to_{pname}"].append(_health)
                _result_dataset.data[f"sensitivity_of_cpt_to_{pname}"].append(
                    getattr(p, 'cpt', 0) if _usable else None)
                if not _usable:
                    _result_dataset.data[f"sensitivity_of_objectives_to_{pname}"].append(None)
                    _result_dataset.data[f"sensitivity_of_ogr_to_{pname}"].append(None)
                    _result_dataset.data[f"sensitivity_of_solutions_to_{pname}"].append({})
                    if _key_vars:
                        _result_dataset.data[f"sensitivity_of_vars_to_{pname}"].append(
                            {vk: None for vk in _key_vars})
                else:
                    try:
                        _obj = None
                        if p.number_of_objectives == 1:
                            _obj = p.objective_values[0][0] if p.objective_values is not None else None
                        else:
                            _obj = p.objective_values
                        _result_dataset.data[f"sensitivity_of_objectives_to_{pname}"].append(_obj)
                    except Exception:
                        _result_dataset.data[f"sensitivity_of_objectives_to_{pname}"].append(None)
                    try:
                        _ogr_val = p.em.get_ogr()
                        _result_dataset.data[f"sensitivity_of_ogr_to_{pname}"].append(_ogr_val)
                    except Exception:
                        _result_dataset.data[f"sensitivity_of_ogr_to_{pname}"].append(None)
                    try:
                        if p.number_of_objectives == 1:
                            _sol = p.solutions if isinstance(p.solutions, dict) else {}
                        else:
                            if isinstance(p.solutions, list) and p.solutions:
                                _sol = {i: dict(sol) for i, sol in enumerate(p.solutions)}
                            elif isinstance(p.solutions, dict):
                                _sol = p.solutions
                            else:
                                _sol = {}
                        _result_dataset.data[f"sensitivity_of_solutions_to_{pname}"].append(_sol)
                    except Exception:
                        _result_dataset.data[f"sensitivity_of_solutions_to_{pname}"].append({})
                    if _key_vars:
                        _var_vals = {}
                        for vk in _key_vars:
                            try:
                                _var_vals[vk] = p.em.get_numpy_var(vk)
                            except Exception:
                                try:
                                    _var_vals[vk] = p.solutions.get(vk) if isinstance(p.solutions, dict) else None
                                except Exception:
                                    _var_vals[vk] = None
                        _result_dataset.data[f"sensitivity_of_vars_to_{pname}"].append(_var_vals)

            _dataset.data[pname] = _original_value

        import itertools as _itertools
        _interaction_data = []
        if len(_key_params) >= 2:
            _base_obj_raw = None
            try:
                if p.number_of_objectives == 1:
                    _base_obj_raw = float(p.objective_values[0][0]) if p.objective_values is not None else None
                else:
                    _base_obj_raw = p.objective_values
            except Exception:
                pass
            _base_obj = ReportEngine._aggregate_objective_static(_base_obj_raw, p.number_of_objectives == 1)

            if p.number_of_objectives == 1:
                _base_sol = p.solutions if isinstance(p.solutions, dict) else {}
            else:
                if isinstance(p.solutions, list) and p.solutions:
                    _base_sol = {i: dict(sol) for i, sol in enumerate(p.solutions)}
                elif isinstance(p.solutions, dict):
                    _base_sol = p.solutions
                else:
                    _base_sol = {}
            _interaction_pairs = list(_itertools.combinations(range(len(_key_params)), 2))

            for _pi, _pj in _interaction_pairs:
                _pname_a = _key_params[_pi]
                _pname_b = _key_params[_pj]
                update_progress(
                    f"Analyzing the interaction of {_pname_a} and {_pname_b}")
                _scenarios_a = _scenarios[_pi]
                _scenarios_b = _scenarios[_pj]
                _ctrl_a = _control_indices.get(_pname_a, 0)
                _ctrl_b = _control_indices.get(_pname_b, 0)

                _max_a = len(_scenarios_a) - 1
                _max_b = len(_scenarios_b) - 1

                _obj_a_key = f"sensitivity_of_objectives_to_{_pname_a}"
                _obj_b_key = f"sensitivity_of_objectives_to_{_pname_b}"
                _sol_a_key = f"sensitivity_of_solutions_to_{_pname_a}"
                _sol_b_key = f"sensitivity_of_solutions_to_{_pname_b}"

                _obj_a_data = _result_dataset.data.get(_obj_a_key, [])
                _obj_b_data = _result_dataset.data.get(_obj_b_key, [])
                _sol_a_data = _result_dataset.data.get(_sol_a_key, [])
                _sol_b_data = _result_dataset.data.get(_sol_b_key, [])

                _z00 = None
                try:
                    if _ctrl_a < len(_obj_a_data) and _obj_a_data[_ctrl_a] is not None:
                        _z00 = ReportEngine._aggregate_objective_static(_obj_a_data[_ctrl_a], p.number_of_objectives == 1)
                except Exception:
                    pass
                if _z00 is None and _base_obj is not None:
                    _z00 = float(_base_obj)

                _z10 = None
                try:
                    if _max_a < len(_obj_a_data) and _obj_a_data[_max_a] is not None:
                        _z10 = ReportEngine._aggregate_objective_static(_obj_a_data[_max_a], p.number_of_objectives == 1)
                except Exception:
                    pass

                _z01 = None
                try:
                    if _max_b < len(_obj_b_data) and _obj_b_data[_max_b] is not None:
                        _z01 = ReportEngine._aggregate_objective_static(_obj_b_data[_max_b], p.number_of_objectives == 1)
                except Exception:
                    pass

                _sol_00 = _base_sol
                _sol_10 = _sol_a_data[_max_a] if _max_a < len(_sol_a_data) and isinstance(_sol_a_data[_max_a], dict) else {}
                _sol_01 = _sol_b_data[_max_b] if _max_b < len(_sol_b_data) and isinstance(_sol_b_data[_max_b], dict) else {}

                _original_a = _copy.deepcopy(_dataset.data[_pname_a])
                _original_b = _copy.deepcopy(_dataset.data[_pname_b])
                _dataset.data[_pname_a] = _rewrap(_original_a, _scenarios_a[_max_a])
                _dataset.data[_pname_b] = _rewrap(_original_b, _scenarios_b[_max_b])

                _z11 = None
                _sol_11 = {}
                _scenario_start = _time.time()
                from ..helpers.formatter import suppress_output
                try:
                    with suppress_output():
                        _clear_stale_results()
                        p.create_env(_env, verbose=True)
                        p.run(verbose=True)
                    # only a healthy combined solve may feed the interaction
                    # term: an unhealthy/infeasible run leaves stale or
                    # meaningless objective values behind
                    if p.objective_values is not None and p.em.healthy():
                        _z11 = ReportEngine._aggregate_objective_static(p.objective_values, p.number_of_objectives == 1)
                        if p.number_of_objectives == 1:
                            if isinstance(p.solutions, dict):
                                _sol_11 = p.solutions
                        else:
                            if isinstance(p.solutions, list) and p.solutions:
                                _sol_11 = {i: dict(sol) for i, sol in enumerate(p.solutions)}
                            elif isinstance(p.solutions, dict):
                                _sol_11 = p.solutions
                except Exception:
                    pass

                _dataset.data[_pname_a] = _original_a
                _dataset.data[_pname_b] = _original_b

                # ============================================================================
                # INTERACTION TERM (I_ab): 2-way ANOVA interaction
                # Formula: I_ab = Z11 - Z10 - Z01 + Z00
                # 
                # Interpretation:
                # - Positive: Synergy (combined > sum of parts)
                # - Negative: Antagonism (combined < sum of parts)
                # - Zero: Additive (independent effects)
                #
                # Theoretical Foundation:
                # Classical factorial design interaction (Fisher, 1935)
                # Widely used in DOE, response surface methodology, and sensitivity analysis
                # 
                # Validation: ✓ Theoretically sound and defensible
                # ============================================================================
                _i_ab = None
                if _z00 is not None and _z10 is not None and _z01 is not None and _z11 is not None:
                    _i_ab = _z11 - _z10 - _z01 + _z00

                _d_10 = ReportEngine._compute_decision_change(_sol_00, _sol_10)
                _d_01 = ReportEngine._compute_decision_change(_sol_00, _sol_01)
                _d_11 = ReportEngine._compute_decision_change(_sol_00, _sol_11)

                _interaction_data.append({
                    'pair': f"{_pname_a} x {_pname_b}",
                    'z00': _z00,
                    'z10': _z10,
                    'z01': _z01,
                    'z11': _z11,
                    'i_ab': _i_ab,
                    'd_00': 0.0,
                    'd_10': _d_10,
                    'd_01': _d_01,
                    'd_11': _d_11,
                })

        p.sensitivity_data = _copy.deepcopy(_result_dataset.data)
        p.sensitivity_interaction_data = _interaction_data
        p.sensitivity_parameter_names = _key_params
        p.sensitivity_parameter_values = _scenarios
        p.sensitivity_control_indices = _control_indices
        p.sensitivity_analyzed = True

        # --- Constraint batch add/remove sensitivity ---------------------
        # Re-solve once per declared constraint batch with that batch
        # excluded.  The two absolutes are the model WITHOUT the batch (the
        # counterfactual), while dObj%/dT% read the other way round: they are
        # measured against that without-model, so a positive number says the
        # constraint EXISTING raises the objective / adds solving time and a
        # negative one says it pulls them down.  (No batches -> no extra runs.)
        p.sensitivity_batch_data = []
        try:
            _batch_registry = ReportEngine._problem_batches(
                p.em.features.get('constraint_batches'))
        except Exception:
            _batch_registry = {}
        if _batch_registry:
            from ..helpers.formatter import suppress_output
            _n_obj = p.number_of_objectives
            _batch_full_obj = ReportEngine._aggregate_objective_static(
                _orig_objective_values, p.number_of_objectives == 1)
            _batch_full_means = ReportEngine._objective_means(
                _orig_objective_values, _n_obj)
            _batch_full_time = _orig_cpt
            for _bname, _bentry in _batch_registry.items():
                update_progress(f"Analyzing batch {_bname}")
                _rec = {'name': _bname,
                        'n': len(_bentry.get('elements', ()) or ()),
                        'obj_full': _batch_full_obj,
                        'obj_full_means': _batch_full_means,
                        'obj_full_pct_k': None,
                        'time_full': _batch_full_time,
                        'obj_without': None, 'obj_delta': None,
                        'obj_pct': None, 'time_without': None,
                        'obj_without_means': None, 'obj_pct_k': None,
                        'time_pct': None, 'status': 'FAIL'}
                try:
                    p._exclude_batches = (_bname,)
                    with suppress_output():
                        _clear_stale_results()
                        p.create_env(_env, verbose=True)
                        p.run(verbose=True)
                    _rec['time_without'] = getattr(p, 'cpt', None)
                    try:
                        _bh = p.em.healthy()
                    except Exception:
                        _bh = None
                    if _bh is True and p.objective_values is not None:
                        _bobj = ReportEngine._aggregate_objective_static(
                            p.objective_values, p.number_of_objectives == 1)
                        _bmeans = ReportEngine._objective_means(
                            p.objective_values, _n_obj)
                        _rec['obj_without'] = _bobj
                        _rec['obj_without_means'] = _bmeans
                        _rec['status'] = 'OK'
                        try:
                            if _batch_full_obj is not None and _bobj is not None:
                                # baseline = the model without the batch, so
                                # +X% reads as "keeping this constraint costs
                                # X% of objective".
                                _d = float(_batch_full_obj) - float(_bobj)
                                _rec['obj_delta'] = _d
                                if abs(float(_bobj)) > 1e-12:
                                    _rec['obj_pct'] = (_d / abs(float(_bobj))
                                                       * 100.0)
                            # same read, kept apart per objective: each
                            # column is the change in that objective's
                            # average across the front
                            if (_batch_full_means is not None
                                    and _bmeans is not None
                                    and len(_batch_full_means) == len(_bmeans)):
                                _pct_k = []
                                for _fk, _wk in zip(_batch_full_means,
                                                    _bmeans):
                                    if _wk is None or abs(float(_wk)) <= 1e-12:
                                        _pct_k.append(None)
                                        continue
                                    _pct_k.append(
                                        (float(_fk) - float(_wk))
                                        / abs(float(_wk)) * 100.0)
                                _rec['obj_pct_k'] = _pct_k
                                _rec['obj_full_pct_k'] = list(_batch_full_means)
                            if (_batch_full_time is not None
                                    and _rec['time_without'] is not None
                                    and float(_rec['time_without']) > 1e-12):
                                _rec['time_pct'] = (
                                    (float(_batch_full_time) - float(_rec['time_without']))
                                    / float(_rec['time_without']) * 100.0)
                        except (TypeError, ValueError):
                            pass
                    elif _bh is False:
                        _rec['status'] = 'INFEAS'
                except Exception:
                    _rec['status'] = 'FAIL'
                finally:
                    p._exclude_batches = None
                p.sensitivity_batch_data.append(_rec)

        p.sensitivity_end_timer = _time.time()

        _dataset.data.clear()
        _dataset.data.update(_copy.deepcopy(_orig_dataset_snapshot))

        if _orig_objective_values is not None:
            p.objective_values = _orig_objective_values
        if _orig_solutions is not None:
            p.solutions = _orig_solutions
        if _orig_em is not None:
            p.em = _orig_em
        if _orig_cpt is not None:
            p.cpt = _orig_cpt
        if _orig_mgt is not None:
            p.mgt = _orig_mgt
        if _orig_data is not None:
            p.data = _orig_data
        for _a in _boost_attrs:
            _v = _orig_boost_attrs.get(_a, _MISSING)
            if _v is _MISSING:
                try:
                    delattr(p, _a)
                except AttributeError:
                    pass
            else:
                setattr(p, _a, _v)

        return p.sensitivity_data

    def report_explain(self, style=1, width=78, box=None):
        import numpy as _np
        p = self._p
        if box is None:
            box = report(width=width, style=style)
        if not hasattr(p, 'em') or p.em is None:
            box.top(left="Explain")
            box.row(left="  No model results available.")
            box.bottom()
            if not box._buffered:
                box.render()
            return

        sd = getattr(p, 'sensitivity_data', None)

        _method = getattr(p, 'method', 'unknown')
        _is_heuristic = _method == 'heuristic'
        _is_madm = _method == 'madm'
        _is_sequential = _method == 'sequential'
        _is_constraint = _method == 'constraint'
        _is_uncertain = _method == 'uncertain'

        if sd is None and not _is_madm and not _is_sequential:
            sd = self._try_auto_sensitivity(p)
        _n_obj = getattr(p, 'number_of_objectives', 0)
        _is_single = _n_obj == 1
        _is_multi = _n_obj > 1

        _n_vars = 0
        _n_cons = 0
        _n_obj_feat = 0
        try:
            _n_vars = p.em.features.get('total_variable_counter', [0, 0])[1]
            _n_cons = p.em.features.get('constraint_counter', [0, 0])[1]
            _n_obj_feat = p.em.features.get('objective_counter', [0, 0])[1]
        except Exception:
            pass

        _directions = getattr(p, 'directions', []) or []
        _interface = getattr(p, 'interface', 'unknown')
        _solver = getattr(p, 'solver', 'unknown')
        _healthy = p.healthy()

        _prog_type = '?'
        _auto_lin = False
        _orig_type = None
        _al_type = None
        try:
            _prog_type = p.em.features.get('problem_type', '?')
            _auto_lin = p.em.features.get('_auto_lin_active', False)
            if _auto_lin:
                _orig_type = p.em.features.get('original_problem_type', '?')
                _al_type = p.em.features.get('al_problem_type', '?')
        except Exception:
            pass

        _has_bvar = False
        _has_ivar = False
        _has_pvar = False
        _has_fvar = False
        try:
            _has_bvar = p.em.features.get('binary_variable_counter', [0, 0])[0] > 0
            _has_ivar = p.em.features.get('integer_variable_counter', [0, 0])[0] > 0
            _has_pvar = p.em.features.get('positive_variable_counter', [0, 0])[0] > 0
            _has_fvar = p.em.features.get('free_variable_counter', [0, 0])[0] > 0
        except Exception:
            pass

        if _is_madm:
            _type_label = "Multi-Attribute Decision Making"
        elif _is_sequential:
            _type_label = "Sequential Decision"
        elif _is_single:
            _type_label = "Single-objective"
        elif _is_multi:
            _type_label = "Multi-objective"
        else:
            _type_label = "Unknown"

        box.top(left="Explain")

        if not _healthy:
            box.row(left="  The model is infeasible or the solver failed.")
            box.row(left="  No valid solution was found.")
        elif sd is None and not _is_madm and not _is_sequential:
            box.row(left="  No sensitivity analysis was performed.")
            box.row(left="  Run sensitivity() or set scenarios for impact assessment.")
        else:
            if _is_single and not _is_madm:
                try:
                    _obj = p.objective_values[0][0] if p.objective_values is not None else None
                    _ogr = p.get_ogr()
                    if _obj is not None:
                        _d = _directions[0] if _directions else 'min'
                        if _ogr is not None and _ogr == 0:
                            box.row(left=f"  The solver found an optimal solution with objective {format_string(_obj)} ({_d}).")
                        elif _ogr is not None and _ogr < 0.01:
                            box.row(left=f"  The solver found a near-optimal solution with objective {format_string(_obj)} ({_d}).")
                        else:
                            box.row(left=f"  The solver found a solution with objective {format_string(_obj)} ({_d}).")
                except Exception:
                    pass

            if _is_multi and not _is_madm:
                try:
                    _n_pareto = 0
                    if hasattr(p, 'solutions') and isinstance(p.solutions, dict):
                        _n_pareto = len(p.solutions)
                    elif hasattr(p, 'num_objective_values'):
                        _n_pareto = p.num_objective_values
                    if _n_pareto > 0:
                        box.row(left=f"  The solver found {_n_pareto} Pareto-optimal solutions.")
                    _n_obj = len(_directions) if _directions else 0
                    if _n_obj > 0:
                        _dir_str = ", ".join([f"{_d}" for _d in _directions[:_n_obj]])
                        box.row(left=f"  Optimizing {_n_obj} objectives: [{_dir_str}].")
                except Exception:
                    pass

            if _is_heuristic:
                try:
                    _repeat = getattr(p, 'repeat', 1)
                    _stg = getattr(p, 'stg', None)
                    if _repeat > 1:
                        box.row(left=f"  The heuristic was run {_repeat} times and the best result was kept.")
                    if _stg is not None and _stg > 0.8:
                        box.row(left=f"  The search stagnated (STG={format_string(_stg)}), suggesting limited exploration.")
                except Exception:
                    pass

            if _is_single and _healthy and not _is_madm:
                try:
                    _sol = p.solutions
                    _key_vars = getattr(p, 'key_vars', []) or []
                    if _sol and isinstance(_sol, dict):
                        _n_active = 0
                        _n_total = 0
                        for k, v in _sol.items():
                            if k.startswith('_') or 'autolin' in k or 'sos2' in k or 'rdiv' in k:
                                continue
                            if _key_vars and k not in _key_vars:
                                continue
                            if isinstance(v, _np.ndarray):
                                _n_total += v.size
                                _n_active += int(_np.count_nonzero(v))
                            elif isinstance(v, (list, tuple)):
                                _n_total += len(v)
                                _n_active += sum(1 for x in v if x != 0)
                            elif isinstance(v, (int, float)):
                                _n_total += 1
                                if v != 0:
                                    _n_active += 1
                        if _n_total > 0 and _n_active > 0:
                            if _n_total == 1:
                                box.row(left=f"  The decision variable is active (non-zero).")
                            else:
                                box.row(left=f"  The solution activates {_n_active} out of {_n_total} decision elements.")
                except Exception:
                    pass

            if _is_multi and _healthy and not _is_madm:
                try:
                    _sol = p.solutions
                    _key_vars = getattr(p, 'key_vars', []) or []
                    if _sol and isinstance(_sol, dict):
                        _n_pareto = len(_sol)
                        _n_active_total = 0
                        _n_total_total = 0
                        for _pidx, _psol in _sol.items():
                            if isinstance(_psol, dict):
                                for k, v in _psol.items():
                                    if k.startswith('_') or 'autolin' in k or 'sos2' in k or 'rdiv' in k:
                                        continue
                                    if _key_vars and k not in _key_vars:
                                        continue
                                    if isinstance(v, _np.ndarray):
                                        _n_total_total += v.size
                                        _n_active_total += int(_np.count_nonzero(v))
                                    elif isinstance(v, (list, tuple)):
                                        _n_total_total += len(v)
                                        _n_active_total += sum(1 for x in v if x != 0)
                                    elif isinstance(v, (int, float)):
                                        _n_total_total += 1
                                        if v != 0:
                                            _n_active_total += 1
                        if _n_pareto > 0 and _n_active_total > 0:
                            box.row(left=f"  {_n_pareto} Pareto solutions activate {_n_active_total} total decision elements.")
                except Exception:
                    pass

        box.bottom()
        if not box._buffered:
            box.render()


from .._version import __version__, __release_month__, __release_year__
