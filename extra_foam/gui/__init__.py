from PyQt5.QtWidgets import QApplication

from .pyqtgraph import setConfigOptions
from .main_gui import MainGUI
from ..config import config


setConfigOptions(
    imageAxisOrder="row-major",
    foreground=config["GUI_FOREGROUND_COLOR"],
    background=config["GUI_BACKGROUND_COLOR"],
)


def mkQApp(args=None):
    app = QApplication.instance()
    if app is None:
        if args is None:
            return QApplication([])
        return QApplication(args)

    return app
