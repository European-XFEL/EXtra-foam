"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from .base_window import _AbstractSatelliteWindow


class AboutWindow(_AbstractSatelliteWindow):
    _title = "About"

    _root_dir = osp.dirname(osp.abspath(__file__))

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._cw = QWidget()
        logger_font = QFont("monospace")
        logger_font.setStyleHint(QFont.TypeWriter)
        logger_font.setPointSize(11)
        self._cw.setFont(logger_font)
        bkg_img_path = osp.join(self._root_dir, '../icons/help_background')
        self._cw.setStyleSheet(f"background-image: url({bkg_img_path})")
        self.setCentralWidget(self._cw)

        self.initUI()

        self.setFixedSize(600, 450)
        self.show()

    def initUI(self):
        """Override."""
        layout = QVBoxLayout()

        name = QLabel("EXtra-foam")
        name_ft = QFont("Helvetica")
        name_ft.setStyleHint(QFont.TypeWriter)
        name_ft.setPointSize(32)
        name_ft.setBold(True)
        name.setFont(name_ft)

        doc_lb = QLabel("<a href = 'https://extra-foam.readthedocs.io/en/stable/'>Documentation</a>")
        doc_ft = QFont("monospace")
        doc_ft.setStyleHint(QFont.TypeWriter)
        doc_ft.setPointSize(14)
        doc_lb.setFont(doc_ft)
        doc_lb.setOpenExternalLinks(True)

        copyright_lb = QLabel(
            "Copyright (C) European X-Ray Free-Electron Laser Facility GmbH. "
            "All rights reserved.")

        layout.addWidget(name)
        layout.addStretch(10)
        layout.addWidget(doc_lb)
        layout.addWidget(copyright_lb)

        self._cw.setLayout(layout)

    def initConnections(self):
        """Override."""
        pass
