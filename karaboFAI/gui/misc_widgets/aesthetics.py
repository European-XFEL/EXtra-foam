"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Hold classes of miscellaneous widgets.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph import ColorMap, intColor, mkPen, mkBrush
from ..pyqtgraph.graphicsItems.GradientEditorItem import Gradients


class QualitativeColors:

    def __init__(self, alpha=255):
        self.n = (251, 154, 153, alpha)  # pink
        self.r = (227, 26, 28, alpha)  # red
        self.o = (255, 127, 0, alpha)  # orange
        self.y = (255, 255, 153, alpha)  # yellow
        self.c = (166, 206, 227, alpha)  # cyan
        self.b = (31, 120, 180, alpha)  # blue
        self.s = (178, 223, 138, alpha)  # grass green
        self.g = (51, 160, 44, alpha)  # green
        self.p = (106, 61, 154, alpha)  # purple
        self.d = (202, 178, 214, alpha)  # orchid
        self.w = (177, 89, 40, alpha)  # brown


Colors = QualitativeColors


def make_pen(color, alpha=255, **kwargs):
    """Convenient function for making QPen."""
    if color is None:
        return mkPen(None)

    # Due to a bug in Qt, setting width greater than 1 will significantly
    # degrade the performance.
    # https://github.com/pyqtgraph/pyqtgraph/issues/533
    width = 1

    if isinstance(color, int):
        return mkPen(intColor(color, **kwargs), width=width)

    return mkPen(getattr(Colors(alpha=alpha), color[0]), width=width, **kwargs)


def make_brush(color, alpha=255):
    return mkBrush(getattr(Colors(alpha=alpha), color[0]))


# Valid keys: thermal, flame, yellowy, bipolar, spectrum, cyclic, greyclip, grey
colorMapFactory = \
    {name: ColorMap(*zip(*Gradients[name]["ticks"]))
     for name in Gradients.keys()}


lookupTableFactory = {name: cmap.getLookupTable()
                      for name, cmap in colorMapFactory.items()}
