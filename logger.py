import logging
import sys


logging.basicConfig(level=logging.INFO,
                    format="PID %(process)5s: %(message)s",
                    stream=sys.stderr)
log = logging.getLogger()


class GuiLogger(logging.Handler):
    def __init__(self, edit):
        super().__init__(level=logging.INFO)
        self._edit = edit

    def emit(self, record):
        self._edit.appendPlainText(self.format(record))
