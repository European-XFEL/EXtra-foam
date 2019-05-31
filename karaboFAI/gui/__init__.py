from PyQt5.QtWidgets import QApplication

from .pyqtgraph import setConfigOptions
from .main_gui import MainGUI

setConfigOptions(
    imageAxisOrder="row-major",
)

__QApp = None


def mkQApp(args=None):
    global __QApp

    __QApp = QApplication.instance()
    if __QApp is None:
        if args is None:
            __QApp = QApplication([])
        else:
            __QApp = QApplication(args)

    return __QApp
