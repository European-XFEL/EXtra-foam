"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QGuiApplication, QCursor
from PyQt5.QtWidgets import QPushButton


def parse_boundary(text):
    """Parse a string which represents the boundary of a value.

    :param str text: the input string.

    :return tuple: (lower boundary, upper boundary).

    Examples:

    parse_boundary("1, 2") == (1, 2)
    """
    msg = "Input lower and upper boundaries separated by comma."
    try:
        if "," not in text:
            raise ValueError(msg)

        # float('Inf') = inf
        ret = [float(x) for x in text.split(",")]

        if len(ret) != 2:
            raise ValueError(msg)

        if ret[0] >= ret[1]:
            raise ValueError("lower boundary >= upper boundary")

    except Exception as e:
        raise ValueError(str(e))

    return ret[0], ret[1]


def parse_id(text):
    """Parse a string into a list of integers.

    :param str text: the input string.

    :return list: a list of IDs which are integers.

    :raise ValueError

    Examples:

    Input IDs separated by comma:

    parse_id("1, 2, 3") == [1, 2, 3]
    parse_id("1, 2, ,3") == [1, 2, 3]
    parse_id("1, 1, ,1") == [1]

    Range-based input:

    parse_id("0:5") == [0, 1, 2, 3, 4]
    parse_id("0:5:2") == [0, 2, 4]

    Combination of the above two styles:

    parse_id("1:3, 5") == [1, 2, 5]

    ":" means all the pulses in a train. The indices list will be generated
    after data is received.

    parse_id(":") == [-1]
    """
    def parse_item(v):
        if not v:
            return []

        if v.strip() == ':':
            return [-1]

        if ':' in v:
            try:
                x = v.split(':')
                if len(x) < 2 or len(x) > 3:
                    raise ValueError("The input is incomprehensible!")

                start = int(x[0].strip())
                if start < 0:
                    raise ValueError("Pulse index cannot be negative!")
                end = int(x[1].strip())

                if len(x) == 3:
                    inc = int(x[2].strip())
                    if inc <= 0:
                        raise ValueError(
                            "Increment must be a positive integer!")
                else:
                    inc = 1

                return list(range(start, end, inc))

            except Exception as e:
                raise ValueError("Invalid input: " + repr(e))

        else:
            try:
                v = int(v)
                if v < 0:
                    raise ValueError("Pulse index cannot be negative!")
            except Exception as e:
                raise ValueError("Invalid input: " + repr(e))

        return v

    ret = set()
    # first split string by comma, then parse them separately
    for item in text.split(","):
        item = parse_item(item.strip())
        if isinstance(item, int):
            ret.add(item)
        else:
            ret.update(item)

    return sorted(ret)


def parse_slice(text):
    """Parse a string into list which can be converted into a slice object.

    :param str text: the input string.

    :return tuple: a list which can be converted into a slice object.

    :raise ValueError

    Examples:

    Input IDs separated by comma:

    parse_slice(":") == [None, None]
    parse_slice("1:2") == [1, 2]
    parse_slice("0:10:2") == [0, 10, 2]
    """
    err_msg = f"Failed to convert '{text}' to a slice object."

    if text:
        parts = text.split(':')
        if len(parts) == 1:
            # slice(stop)
            parts = [None, parts[0]]
        # else: slice(start, stop[, step])
    else:
        raise ValueError(err_msg)

    try:
        ret = [int(p) if p else None for p in parts]
        if len(ret) > 3:
            raise ValueError(err_msg)
        return ret
    except Exception:
        raise ValueError(err_msg)


def parse_slice_inv(text):
    """Parse a string into a slice notation.

    This function inverts the result from 'parse_slice'.

    :param str text: the input string.

    :return str: the slice notation.

    :raise ValueError

    Examples:

    parse_slice_inv('[None, None]') == ":"
    parse_slice_inv('[1, 2]') == "1:2"
    parse_slice_inv('[0, 10, 2]') == "0:10:2"
    """
    err_msg = f"Failed to convert '{text}' to a slice notation."

    if len(text) > 1:
        try:
            parts = [None if v.strip() == 'None' else int(v)
                     for v in text[1:-1].split(',')]
        except ValueError:
            raise ValueError(err_msg)

        if len(parts) == 2:
            s0 = '' if parts[0] is None else str(parts[0])
            s1 = '' if parts[1] is None else str(parts[1])
            return f"{s0}:{s1}"

        if len(parts) == 3:
            s0 = '' if parts[0] is None else str(parts[0])
            s1 = '' if parts[1] is None else str(parts[1])
            s2 = '' if parts[2] is None else str(parts[2])
            return f"{s0}:{s1}:{s2}"

    raise ValueError(err_msg)


def get_icon_path(filename):
    root_dir = osp.dirname(osp.abspath(__file__))
    return osp.join(root_dir, "icons", filename)

def create_icon_button(filename, size, *, description=""):
    """Create a QPushButton with icon.

    :param str filename: name of the icon file.
    :param int size: size of the icon (button).
    :param str description: tool tip of the button.
    """
    btn = QPushButton()
    icon = QIcon(get_icon_path(filename))
    btn.setIcon(icon)
    btn.setIconSize(QSize(size, size))
    btn.setFixedSize(btn.minimumSizeHint())
    if description:
        btn.setToolTip(description)
    return btn


def invert_dict(mapping):
    """Return a dictionary with key and value swapped."""
    ret = dict()
    for k, v in mapping.items():
        ret[v] = k
    return ret

def center_window(window, resize=True):
    """Center the window to the screen the cursor is placed on.

    Optionally also resize the window so that it fits on the screen.

    :param QWidget window: Window widget to position.
    :param bool    resize: Whether or not to resize the window.
    """
    screen = QGuiApplication.screenAt(QCursor.pos())

    if resize:
        screen_size = screen.size()
        max_width = screen_size.width()
        max_height = screen_size.height()

        window.resize(int(max_width * 0.8), int(max_height * 0.8))

    window.move(screen.geometry().center() - window.frameGeometry().center())
