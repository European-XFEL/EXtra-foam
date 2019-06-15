from PyQt5.QtWidgets import QApplication

from .pyqtgraph import setConfigOptions
from .main_gui import MainGUI


setConfigOptions(
    imageAxisOrder="row-major",
    foreground="k",
    background=(225, 225, 225, 255),
)


def mkQApp(args=None):
    app = QApplication.instance()
    if app is None:
        if args is None:
            return QApplication([])
        return QApplication(args)

    return app
