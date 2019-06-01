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


class Colors:
    def __init__(self, alpha=255):
        self.r = (215, 25, 28, alpha)  # red
        self.o = (253, 174, 97, alpha)  # orange
        self.y = (255, 255, 191, alpha)  # yellow
        self.c = (171, 217, 233, alpha)  # cyan
        self.b = (44, 123, 182, alpha)  # blue
        self.g = (26, 150, 65, alpha)  # green
        self.p = (94, 60, 153, alpha)  # purple
        self.d = (218, 112, 214, alpha)  # orchid
        self.w = (247, 247, 247, alpha)  # white
        self.e = (186, 186, 186, alpha)  # dark grey


def make_pen(color, width=2, alpha=255, **kwargs):
    """Convenient function for making QPen."""
    if color is None:
        return mkPen(None)

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
