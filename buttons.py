import logging
from abc import ABC, abstractmethod

from PyQt5.QtWidgets import QPushButton


def get_style_sheet(bg_color):
    """Return the string for setStyleSheet().

    :param bg_color: string
        Color Hex Color Codes.
    """
    style_sheet = 'QPushButton {color: white; font: bold; padding: 5px; ' \
                  + 'background-color: ' + bg_color + '}'
    return style_sheet


class PlotButton(QPushButton):
    """Inherited from QPushButton."""
    def __init__(self, name, parent=None, size=(160, 40)):
        """Initialization."""
        super().__init__(name, parent)
        self.setStyleSheet(get_style_sheet("#610B4B"))
        self.setFixedSize(*size)


class RunButton(QPushButton):
    """Inherited from QPushButton."""
    def __init__(self, name='Run', parent=None, size=(80, 20)):
        """Initialization"""
        super().__init__(name, parent)
        self.setFixedSize(*size)

        self.observers = []

        self._state = self._create_state("stop")
        self._state.on_enter(self)

    def attach(self, p):
        self.observers.append(p)

    def update(self):
        self._state.on_exit()
        self._state = self._state.next()
        self._state.on_enter(self)

    def _create_state(self, name):
        """"""
        class DataAcquisitionState(ABC):
            @abstractmethod
            def on_enter(self, btn):
                """Action when entering this state.

                :param btn: QPushButton object
                    A QPushButton instance.
                """
                pass

            @abstractmethod
            def on_exit(self):
                pass

            @abstractmethod
            def next(self):
                pass

        class DataAcquisitionStart(DataAcquisitionState):
            def on_exit(self):
                pass

            def on_enter(self, btn):
                btn.setText("Stop")
                btn.setStyleSheet("background-color: Red")
                logging.warning("Start acquiring data...")
                for widget in btn.observers:
                    widget.is_running = True

            def next(self):
                return DataAcquisitionStop()

        class DataAcquisitionStop(DataAcquisitionState):
            def on_exit(self):
                pass

            def on_enter(self, btn):
                btn.setText("Run")
                btn.setStyleSheet("background-color: Green")
                logging.warning("Stop acquiring data!")
                for widget in btn.observers:
                    widget.is_running = False

            def next(self):
                return DataAcquisitionStart()

        if name == "start":
            return DataAcquisitionStart()

        if name == "stop":
            return DataAcquisitionStop()

        raise ValueError("Unknown data acquisition state!")
