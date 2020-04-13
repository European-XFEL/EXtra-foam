"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import QObject

from ..config import config
from ..geometries import load_geometry, GeomAssembler


class GeometryItem(QObject):
    __instance = None

    def __new__(cls, *args, **kwargs):
        """Create a singleton."""
        if cls.__instance is None:
            cls.__instance = super().__new__(cls, *args, **kwargs)
            cls.__instance._is_initialized = False
        return cls.__instance

    def __init__(self, *args, **kwargs):
        if self._is_initialized:
            return

        super().__init__(*args, **kwargs)

        self._detector = config["DETECTOR"]

        self._geom = None
        self._update = True

        self._filepath = ""
        self._assembler = GeomAssembler.OWN
        self._quad_positions = None
        self._stack_only = False

    def setFilepath(self, v):
        if v != self._filepath:
            self._filepath = v
            self._update = True

    def setAssembler(self, v):
        if v != self._assembler:
            self._assembler = v
            self._update = True

    def setQuadPositions(self, v):
        if v != self._quad_positions:
            self._quad_positions = v
            self._update = True

    def setStackOnly(self, v):
        if v != self._stack_only:
            self._stack_only = v
            self._update = True

    @property
    def geometry(self):
        if self._update:
            self._geom = load_geometry(self._detector, self._filepath,
                                       assembler=self._assembler,
                                       quad_positions=self._quad_positions,
                                       stack_only=self._stack_only)

            self._update = False
        return self._geom