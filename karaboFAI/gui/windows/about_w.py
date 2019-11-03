"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget

from .base_window import AbstractSatelliteWindow


class AboutWindow(AbstractSatelliteWindow):
    title = "About"

    _root_dir = osp.dirname(osp.abspath(__file__))

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._cw = QWidget()
        logger_font = QtGui.QFont("monospace")
        logger_font.setStyleHint(QtGui.QFont.TypeWriter)
        logger_font.setPointSize(11)
        self._cw.setFont(logger_font)
        bkg_img_path = osp.join(self._root_dir, '../icons/help_background')
        self._cw.setStyleSheet(f"background-image: url({bkg_img_path})")
        self.setCentralWidget(self._cw)

        self.initUI()

        self.setFixedSize(600, 450)
        self.show()

    def initUI(self):
        layout = QtGui.QVBoxLayout()

        name = QtGui.QLabel("karaboFAI")
        name_ft = QtGui.QFont("Helvetica")
        name_ft.setStyleHint(QtGui.QFont.TypeWriter)
        name_ft.setPointSize(32)
        name_ft.setBold(True)
        name.setFont(name_ft)

        doc_lb = QtGui.QLabel("<a href = 'https://in.xfel.eu/readthedocs/docs/karabofai/en/latest/'>Documentation</a>")
        doc_ft = QtGui.QFont("monospace")
        doc_ft.setStyleHint(QtGui.QFont.TypeWriter)
        doc_ft.setPointSize(14)
        doc_lb.setFont(doc_ft)
        doc_lb.setOpenExternalLinks(True)

        copyright_lb = QtGui.QLabel(
            "Copyright (C) European X-Ray Free-Electron Laser Facility GmbH. "
            "All rights reserved.")

        layout.addWidget(name)
        layout.addStretch(10)
        layout.addWidget(doc_lb)
        layout.addWidget(copyright_lb)

        self._cw.setLayout(layout)
