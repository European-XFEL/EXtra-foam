"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Helper functions.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import ast

import numpy as np


def parse_boundary(text):
    try:
        lb, ub = [ast.literal_eval(x.strip()) for x in text.split(",")]
    except Exception:
        raise ValueError("Invalid input!")

    if lb > ub:
        raise ValueError("lower boundary > upper boundary!")

    return lb, ub


def parse_ids(text):
    def parse_item(v):
        if not v:
            return []

        if ':' in v:
            x = v.split(':')
            if len(x) < 2 or len(x) > 3:
                raise ValueError
            try:
                start = int(x[0].strip())
                if start < 0:
                    raise ValueError("Pulse ID cannot be negative!")
                end = int(x[1].strip())
                if end <= start:
                    raise ValueError

                if len(x) == 3:
                    inc = int(x[2].strip())
                else:
                    inc = 1
            except ValueError:
                raise

            return list(range(start, end, inc))

        try:
            v = int(v)
            if v < 0:
                raise ValueError
        except ValueError:
            raise ValueError

        return v

    ret = set()
    for item in text.split(","):
        item = parse_item(item.strip())
        if isinstance(item, int):
            ret.add(item)
        else:
            ret.update(item)

    return list(ret)


def parse_quadrant_table(widget):
    n_row, n_col = widget.rowCount(), widget.columnCount()
    ret = np.zeros((n_row, n_col))
    for i in range(n_row):
        for j in range(n_col):
            ret[i, j] = float(widget.item(i, j).text())
    return ret
