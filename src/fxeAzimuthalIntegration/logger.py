import logging
import sys


logging.basicConfig(
    level=logging.DEBUG,
    format="%(filename)s: %(levelname)s: %(message)s",
    stream=sys.stderr)
log = logging.getLogger(__name__)
# disable DEBUG information from imported module pyFAI
logging.getLogger("pyFAI").setLevel(logging.CRITICAL)


class GuiLogger(logging.Handler):
    def __init__(self, edit):
        super().__init__(level=logging.INFO)
        self._edit = edit

    def emit(self, record):
        self._edit.appendPlainText(self.format(record))
