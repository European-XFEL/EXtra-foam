import time

from extra_foam.gui import mkQApp


app = mkQApp()

# For debug
_VISUALIZE = False


def _display(interval=0.5):
    if _VISUALIZE:
        app.processEvents()
        time.sleep(interval)
        return True
    return False

