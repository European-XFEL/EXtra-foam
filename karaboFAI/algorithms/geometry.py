"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Geometry algorithms.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""


def intersection(w1, h1, x1, y1, w2, h2, x2, y2):
    """Calculate the intersection area of two rectangles.

    :param: float w1, w2: width
    :param: float h1, h2: height
    :param: float x1, x2: x coordinate of the closest corner to the origin.
    :param: float y1, y2: y coordinate of the closest corner to the origin.

    :returns tuple: (w, h, x, y) of the intersection area.
    """
    x = max(x1, x2)
    xx = min(x1 + w1, x2 + w2)
    y = max(y1, y2)
    yy = min(y1 + h1, y2 + h2)

    w = xx - x
    h = yy - y

    return w, h, x, y