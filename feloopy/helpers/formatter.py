# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from rich.console import Console
from rich.spinner import Spinner
from datetime import datetime
import threading
import time
import time
import threading
from tqdm import tqdm
import numpy as np
from numbers import Number
from typing import Any, Sequence
import pandas as pd

class report:

    def __init__(
        self,
        topleft="Result",
        topcenter="",
        topright="",
        bottomleft="",
        bottomcenter="",
        bottomright="",
        width=88,
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

    def top(self, left="", center="", right=""):

        len_left = len(left)
        len_right = len(right)
        len_center = len(center)

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

        padding_left = self._topbar * (remaining // 2)
        padding_right = self._topbar * (remaining - len(padding_left))

        print(
            f"{self.start_top}{final_left_text}{padding_left}{final_center_text}{padding_right}{final_right_text}{self.end_top}"
        )

    def bottom(self, left="", center="", right=""):

        len_left = len(left)
        len_right = len(right)
        len_center = len(center)

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

        padding_left = self._bottombar * (remaining // 2)
        padding_right = self._bottombar * (remaining - len(padding_left))

        print(
            f"{self.start_bottom}{final_left_text}{padding_left}{final_center_text}{padding_right}{final_right_text}{self.end_bottom}"
        )

    def empty(self):

        print(self._border + (self.width - 2) * " " + self._border)

    def middle(self, left="", center="", right=""):

        len_left = len(left)
        len_right = len(right)
        len_center = len(center)

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

        padding_left = self._middlebar * (remaining // 2)
        padding_right = self._middlebar * (remaining - len(padding_left))

        print(
            f"{self.start_middle}{final_left_text}{padding_left}{final_center_text}{padding_right}{final_right_text}{self.end_middle}"
        )
    
    
    def row(self, left="", center="", right="", fill_char=False):
        len_left = len(left)
        len_right = len(right)
        len_center = len(center)

        left_space = " " if len_left > 0 else ""
        right_space = " " if len_right > 0 else ""
        center_space = " " if len_center > 0 else ""

        final_left_text = left_space + left + left_space
        final_right_text = right_space + right + right_space
        final_center_text = center_space + center + center_space

        used_width = (
            len(final_left_text) +
            len(final_center_text) +
            len(final_right_text)
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

        print(
            f"{self._border}{final_left_text}{padding_left}{final_center_text}{padding_right}{final_right_text}{self._border}"
        )


    def columns(self, list_of_strings, label, max_space_between_elements=4):
        max_string_length = max(len(s) for s in list_of_strings)
        total_width = self.width

        label_width = len(label)
        remaining_width = (
            total_width
            - label_width
            - 4 * len(list_of_strings)
            - len(list_of_strings) * max_string_length
        )
        min_space_between_elements = 1

        if len(list_of_strings) > 1:
            min_space_between_elements = remaining_width // (len(list_of_strings) - 1)

        space_between_elements = min(
            max_space_between_elements, min_space_between_elements
        )
        if space_between_elements < max_space_between_elements:
            extra_space = remaining_width - space_between_elements * (
                len(list_of_strings) - 1
            )
        else:
            extra_space = 0

        rowstart = f"{self.start_middle} {label} {' ' * extra_space}"
        row = ""
        for string in list_of_strings[:-1]:
            row += f"{' ' * (max_string_length - len(string))}{string}{' ' * space_between_elements}"
        row += f"{' ' * (max_string_length - len(list_of_strings[-1]))}{list_of_strings[-1]}"
        remaining = self.width - len(rowstart) - len(row) - 3
        row = rowstart + remaining * self._middlebar + " " + row
        print(f"{row} │")


    def clear_columns(
        self,
        list_of_strings: Sequence[str],
        label: str,
        max_space_between_elements: int = 4,
        extra_spaces: int = 0
    ) -> None:
        if extra_spaces < 0:
            raise ValueError("extra_spaces must be >= 0")
        max_len = max(len(s) for s in list_of_strings)
        label_part = f"{self._border} {label}"
        tail = " "+self._border
        used = len(label_part) + len(tail) + len(list_of_strings) * max_len
        slots = max(len(list_of_strings) - 1, 1)
        base_space = (self.width - used - extra_spaces * slots) // slots
        space_between = min(base_space, max_space_between_elements)
        leftover = self.width - used - slots * (space_between + extra_spaces)
        left = label_part + " " * leftover
        gap = space_between + extra_spaces
        body = ""
        for i, s in enumerate(list_of_strings):
            body += s.rjust(max_len)
            if i < len(list_of_strings) - 1:
                body += " " * gap
        print(f"{left}{body}{tail}")

    def _format_value(self, val: Any, decimal_places: int) -> str:
        if isinstance(val, Number):
            fv = float(val)
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

    def print_element(
        self,
        label: str,
        var: Any,
        additional_text: str = "",
        decimal_places: int = 4
    ) -> None:
        if isinstance(var, np.ndarray):
            self._print_numpy_element(label, var, decimal_places)
        elif isinstance(var, dict):
            self._print_dict_element(label, var, decimal_places)
        else:
            raise ValueError("print_element only accepts np.ndarray or dict")
        if additional_text:
            print(
                f"{self._border} "
                f"{additional_text.ljust(self.width - 4)} "
                f"{self._border}"
            )

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

            print(f"{self._border} {line[:inner_width].ljust(inner_width)} {self._border}")
        else:
            raise ValueError("print_value only accepts np.ndarray, dict, int, float, or str")

        if additional_text:
            print(
                f"{self._border} "
                f"{additional_text[:inner_width].ljust(inner_width)} "
                f"{self._border}"
            )


    def _print_numpy_element(
        self,
        label: str,
        arr: np.ndarray,
        decimal_places: int
    ) -> None:
        idx = [0] * arr.ndim
        def recurse(dim: int = 0):
            if dim < arr.ndim:
                for i in range(arr.shape[dim]):
                    idx[dim] = i
                    recurse(dim + 1)
            else:
                val = arr[tuple(idx)]
                if float(val) != 0.0:
                    index_str = "[" + ",".join(map(str, idx)) + "]"
                    content = f"{label}{index_str} = {self._format_value(val, decimal_places)}"
                    print(f"{self._border} {content.ljust(self.width - 4)} {self._border}")
        recurse()

    def _print_dict_element(
        self,
        label: str,
        d: dict,
        decimal_places: int
    ) -> None:
        for k, v in d.items():
            if float(v) == 0.0:
                continue
            key_str = str(k).replace("(", "[").replace(")", "]")
            if "[" not in key_str:
                key_str = f"[{key_str}]"
            content = f"{label}{key_str} = {self._format_value(v, decimal_places)}"
            print(f"{self._border} {content.ljust(self.width - 4)} {self._border}")

    def print_tensor(
        self,
        label: str,
        tensor: Any,
        additional_text: str = "",
        decimal_places: int = 4
    ) -> None:
        arr = np.array(tensor)
        def custom_formatter(x):
            fv = float(x)
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
        txt = np.array2string(arr, separator=", ", prefix="", formatter=formatter)
        rows = txt.split("\n")
        for i, row in enumerate(rows):
            if i == 0:
                line = f"{label} = {row}"
                print(
                    f"{self._border} "
                    f"{line.ljust(self.width - 4 - len(additional_text))}"
                    f"{additional_text}"
                    f" {self._border}"
                )
            else:
                indent = " " * (len(label) + 3)
                print(
                    f"{self._border} "
                    f"{(indent + row).ljust(self.width - 4)} "
                    f"{self._border}"
                )

    def print_pandas_df(
        self,
        label: str,
        df: pd.DataFrame,
        columns: Sequence[str] = None,
        additional_text: str = "",
        decimal_places: int = 4,
        max_space_between_elements: int = 4,
        extra_spaces: int = 0
    ) -> None:
        cols = columns if columns is not None else df.columns
        for idx, row in df[cols].iterrows():
            strs = [self._format_value(val, decimal_places) for val in row]
            self.clear_columns(
                strs,
                f"{label}[{idx}]",
                max_space_between_elements=max_space_between_elements,
                extra_spaces=extra_spaces
            )
        if additional_text:
            print(
                f"{self._border} "
                f"{additional_text.ljust(self.width - 4)} "
                f"{self._border}"
            )

def left_align(input, box_width=88, rt=False):

    if rt:
        return "│" + " " + input.ljust(box_width - 2) + " " + "│"
    else:
        print("│" + " " + input.ljust(box_width - 2) + " " + "│")


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


def center(input, box_width=88):
    print("│" + " " + str(input).center(box_width - 2) + " " + "│")


def tline(box_width=88):
    print("┌" + "─" * box_width + "┐")


def tline_text(input, box_width=88):
    print("┌─ " + input + " " + "─" * (box_width - len(input) - 3) + "┐")


def bline_text(input, box_width=88):
    print("└" + input + " " + "─" * (box_width - len(input) - 3) + "┘")


def empty_line(box_width=88):
    print("│" + " " * box_width + "│")


def bline(box_width=88):
    print("└" + "─" * box_width + "┘")


def hline(box_width=88):
    print("├" + "─" * box_width + "┤")


def whline(box_width=88):
    print("+" + "." * box_width + "┤")


def hrule(box_width=88):
    print("├" + "=" * box_width + "┤")


def vspace():
    print()


def two_column(input1, input2, box_width=88):
    padding = box_width - len(input1) - len(input2) - 2
    print("│ " + str(input1) + " " * padding + str(input2) + " │")


def three_column(input1, input2, input3, box_width=88, underline=False):
    total_padding = box_width - len(input1) - len(input2) - len(input3) - 6
    padding1 = total_padding // 2
    padding2 = total_padding - padding1

    if underline:
        input1 = f"\033[4m{input1}\033[0m"
        input2 = f"\033[4m{input2}\033[0m"
        input3 = f"\033[4m{input3}\033[0m"

    print(
        "│ "
        + str(input1)
        + " " * padding1
        + "  "
        + str(input2)
        + " " * padding2
        + "  "
        + str(input3)
        + " │"
    )


def list_three_column(input_list, box_width=88, underline=False):
    column1_width = max(len(str(x[0])) for x in input_list) + 2
    column2_width = max(len(str(x[1])) for x in input_list) + 2
    column3_width = max(len(str(x[2])) for x in input_list) + 2

    space_left = box_width - column1_width - column2_width - column3_width - 2
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
                f"│ {input1:<{column1_width}}{padding1 * ' '}{input2:<{column2_width}}{padding2 * ' '}{input3:<{column3_width}} │"
            )


def boxed(text, box_width=88):
    import textwrap
    lines = textwrap.wrap(text, width=box_width - 4)
    for line in lines:
        left_align(line, box_width)

def right_align(input, box_width=88):
    print("│" + " " + input.rjust(box_width - 2) + " " + "│")

def feature_print(type, input):
    if input[0] > 0:
        three_column(type, f"{input[0]}", f"{input[1]}")


def objective_print(type, input):
    if input[0] > 0:
        three_column(type, "-", f"{input[1]}")


def constraint_print(type, input):
    if input[0] > 0:
        three_column(type, f"{input[0]}", f"{input[1]}")


def status_row_print(ObjectivesDirections, status, box_width=88):
    if len(ObjectivesDirections) != 1 and ObjectivesDirections[0] != "nan":
        row = (
            "│ "
            + "Status: "
            + " " * (len(status[0]) - len("Status: "))
            + " "
            * (box_width - 10 * len(ObjectivesDirections) + 1 - len(str(status[0])) - 3)
        )
        for j in range(len(ObjectivesDirections)):
            obj_row = ObjectivesDirections[j]
            row += " " * (10 - len(obj_row)) + obj_row
        print(row + " │")

    else:
        row = (
            "│ "
            + "Status: "
            + " " * (len(str(status)) - len("Status: "))
            + " "
            * (box_width - 10 * len(ObjectivesDirections) + 1 - len(str(status)) - 3)
        )
        for j in range(len(ObjectivesDirections)):
            obj_row = ObjectivesDirections[j]
            row += " " * (10 - len(obj_row)) + obj_row

        if len(row + " │") == box_width + 2:
            print(row + " │")

        elif len(row + " │") < box_width + 2:
            row = (
                "│ "
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
            print(row + " │")

        else:
            row = (
                "│ "
                + "Status: "
                + " " * (len(str(status)) - len("Status: "))
                + " "
                * (box_width - 10 * len(ObjectivesDirections) - len(str(status)) - 3)
            )
            for j in range(len(ObjectivesDirections)):
                obj_row = ObjectivesDirections[j]
                row += " " * (10 - len(obj_row)) + obj_row
            print(row + " │")


def solution_print(
    ObjectivesDirections, status, get_obj, get_payoff=None, box_width=88
):

    if len(ObjectivesDirections) != 1:
        if status[0] != "infeasible (constrained)":
            for i in range(len(status)):
                row = (
                    "│ "
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
                print(row + " │")

            for j in range(len(ObjectivesDirections)):
                row = (
                    "│ "
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
                print(row + " │")

            row = (
                "│ "
                + str("max")
                + " "
                * (box_width - 10 * len(ObjectivesDirections) + 1 - len(str("max")) - 3)
            )
            for j in range(len(ObjectivesDirections)):
                num_str = format_string(np.max(get_obj[:, j]))
                row += " " * (10 - len(num_str)) + num_str
            print(row + " │")

            row = (
                "│ "
                + str("ave")
                + " "
                * (box_width - 10 * len(ObjectivesDirections) + 1 - len(str("ave")) - 3)
            )
            for j in range(len(ObjectivesDirections)):
                num_str = format_string(np.average(get_obj[:, j]))
                row += " " * (10 - len(num_str)) + num_str
            print(row + " │")

            row = (
                "│ "
                + str("std")
                + " "
                * (box_width - 10 * len(ObjectivesDirections) + 1 - len(str("std")) - 3)
            )
            for j in range(len(ObjectivesDirections)):
                num_str = format_string(np.std(get_obj[:, j]))
                row += " " * (10 - len(num_str)) + num_str
            print(row + " │")

            row = (
                "│ "
                + str("min")
                + " "
                * (box_width - 10 * len(ObjectivesDirections) + 1 - len(str("min")) - 3)
            )
            for j in range(len(ObjectivesDirections)):
                num_str = format_string(np.min(get_obj[:, j]))
                row += " " * (10 - len(num_str)) + num_str
            print(row + " │")

    else:
        row = (
            "│ "
            + str(status)
            + " "
            * (box_width - 9 * len(ObjectivesDirections) + 1 - len(str(status)) - 3)
        )
        obj_row = get_obj
        num_str = format_string(obj_row)
        row += " " * (9 - len(num_str)) + num_str
        print(row + " │")


def metrics_print(
    ObjectivesDirections,
    show_all_metrics,
    get_obj,
    calculated_indicators,
    start=0,
    end=0,
    length=None,
    box_width=88,
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


def run_with_progress(func, show_log, *args, **kwargs):
    def show_progress():
        with tqdm(total=0, unit="s", bar_format="{desc}") as pbar:
            while not stop_event.is_set():
                pbar.set_description("Processing...")
                pbar.update()
                time.sleep(0.5)

    stop_event = threading.Event()

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
    from tqdm import tqdm

    leave = False if remain == False else True
    return tqdm(iterable, desc=description, unit=unit, ncols=82, leave=leave)

_console = Console()
_spinner_thread = None
_spinner_running = False
_is_notebook = False
_start_time = None
_show_elapsed = False


def _detect_notebook():
    try:
        from IPython import get_ipython
        if "IPKernelApp" in get_ipython().config:
            return True
    except Exception:
        pass
    return False


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


def start_progress(message="Processing...", spinner="dots", show_elapsed=False):
    global _spinner_running, _spinner_thread, _is_notebook, _start_time, _show_elapsed

    _is_notebook = _detect_notebook()
    _start_time = datetime.now()
    _show_elapsed = show_elapsed

    def format_message():
        elapsed_str = ""
        if _show_elapsed:
            elapsed_str = f" (Elapsed: {_format_elapsed(datetime.now() - _start_time)})"
        return f"{message}{elapsed_str}"

    def spinner_task():
        if _is_notebook:
            from IPython.display import clear_output
            from rich.jupyter import print as jupyter_print
            while _spinner_running:
                clear_output(wait=True)
                jupyter_print(Spinner(spinner, text=format_message()))
                time.sleep(0.1)
            clear_output(wait=True)
        else:
            status = None
            try:
                status = _console.status(format_message(), spinner=spinner)
                status.start()
                while _spinner_running:
                    status.update(format_message())
                    time.sleep(0.1)
            except LiveError:
                while _spinner_running:
                    _console.print(".", end="", soft_wrap=True)
                    time.sleep(0.5)
                _console.print()
            finally:
                if status:
                    try:
                        status.stop()
                    except Exception:
                        pass

    _spinner_running = True
    _spinner_thread = threading.Thread(target=spinner_task, daemon=True)
    _spinner_thread.start()


def end_progress(success_message="Done!", failure_message=None, success=True, show_elapsed=None):
    global _spinner_running, _spinner_thread

    _spinner_running = False
    if _spinner_thread is not None:
        _spinner_thread.join()

    if _is_notebook:
        from IPython.display import clear_output
        clear_output(wait=True)

    final_show = _show_elapsed if show_elapsed is None else show_elapsed
    elapsed_str = ""
    if final_show and _start_time:
        elapsed_str = f" for {_format_elapsed(datetime.now() - _start_time)}"

    if success:
        _console.print(f"[bold green]{success_message}{elapsed_str}[/bold green]")
    elif failure_message:
        _console.print(f"[bold red]{failure_message}{elapsed_str}[/bold red]")
def calculate_time_difference(start=0, end=0, length=None):
    if length is None:
        delta = round((end - start), 3)
    else:
        delta = round(length, 3)

    hour = delta % (24 * 3600) // 3600
    minute = delta % (24 * 3600) % 3600 // 60
    second = delta % (24 * 3600) % 3600 % 60

    return hour, minute, second


def format_time_and_microseconds(seconds_value, name=""):
    microseconds_value = seconds_value * 1e6
    microseconds_scientific_notation = "{:.2e}".format(microseconds_value)

    hours = int(microseconds_value // 3600e6)
    minutes = int((microseconds_value % 3600e6) // 60e6)
    seconds = int((microseconds_value % 60e6) / 1e6)

    time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return name + f"{time_formatted} h:m:s {microseconds_scientific_notation} μs"