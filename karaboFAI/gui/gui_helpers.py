"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Helper functions for GUI.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""


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


def parse_ids(text):
    """Parse a string into a list of integers.

    :param str text: the input string.

    :return list: a list of IDs which are integers.

    :raise ValueError

    Examples:

    Input IDs separated by comma:

    parse_ids("1, 2, 3") == [1, 2, 3]
    parse_ids("1, 2, ,3") == [1, 2, 3]
    parse_ids("1, 1, ,1") == [1]

    Range-based input:

    parse_ids("0:5") == [0, 1, 2, 3, 4]
    parse_ids("0:5:2") == [0, 2, 4]

    Combination of the above two styles:

    parse_ids("1:3, 5") == [1, 2, 5]
    """
    def parse_item(v):
        if not v:
            return []

        if ':' in v:
            try:
                x = v.split(':')
                if len(x) < 2 or len(x) > 3:
                    raise ValueError("The input is incomprehensible!")

                start = int(x[0].strip())
                if start < 0:
                    raise ValueError("Pulse ID cannot be negative!")
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
                    raise ValueError("Pulse ID cannot be negative!")
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


def parse_table_widget(widget):
    """Parse a table widget to a list of list.

    The inner list represents a row of the table.

    :param QTableWidget widget: a table widget.

    :return list: a list of table elements.

    Examples:

    For the following table,

         col1 col2
    row1   1   2
    row2   3   4
    row3   5   6

    The return value is [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].

    TODO: add test
    """
    n_row, n_col = widget.rowCount(), widget.columnCount()
    ret = []
    for i in range(n_col):
        ret.append([float(widget.item(j, i).text()) for j in range(n_row)])
    return ret
