"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import (
    pyqtSignal, pyqtSlot, QObject, Qt, QThread, QTimer
)
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QAction, QFrame, QHBoxLayout, QMainWindow, QScrollArea, QSplitter,
    QTabWidget, QVBoxLayout, QWidget
)

from extra_foam.gui import mkQApp
from extra_foam import __version__


class GuiExtension(QMainWindow):
    def __init__(self):
        super().__init__()

        self.title = f"EXtra-foam {__version__}"
        self.setWindowTitle(self.title + " - GUI Extension")

        # TODO: Add a Timer to automatically shutdown itself

        self.show()


def gui_extension():
    app = mkQApp()
    gui = GuiExtension()
    app.exec_()


if __name__ == "__main__":
    gui_extension()
