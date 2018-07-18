import pyqtgraph as pg


PLOT_WIDTH = 800
PLOT_HEIGHT = 400

UPDATE_FREQUENCY = 10


class LinePlotWidget(pg.GraphicsLayoutWidget):
    def __init__(self, w, h, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setFixedSize(w, h)

        self.p1 = self.addPlot()
        self.nextRow()
        self.p2 = self.addPlot()


class ImageViewWidget(pg.GraphicsLayoutWidget):
    def __init__(self, w, h, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setFixedSize(w, h)

        self._img = pg.ImageItem(border='w')
        self._view = self.addViewBox()
        self._view.addItem(self._img)

    def set_image(self, data):
        self._img.clear()
        self._img.setImage(data)
        self._view.autoRange()
