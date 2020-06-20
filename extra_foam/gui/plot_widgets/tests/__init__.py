import time

from extra_foam.gui import mkQApp


app = mkQApp()

# For debug
_VISUALIZE = False


def _display():
    if _VISUALIZE:
        app.processEvents()
        time.sleep(1)
        return True
    return False

