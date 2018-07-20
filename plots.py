from PyQt5.QtWidgets import QMainWindow, QWidget, QGridLayout
import numpy as np

import pyqtgraph as pg
import config as cfg


UPDATE_FREQUENCY = 10


class LinePlotWidget(pg.GraphicsLayoutWidget):
    """A widget has multiple PlotItems stacked vertically."""
    def __init__(self, n_items, *, show_image=False, **kwargs):
        """Initialization.

        :param int n_lines: Number of PlotItems.
        """
        super().__init__(**kwargs)

        w = cfg.LINE_PLOT_WIDTH + show_image*(cfg.LINE_PLOT_HEIGHT - 40)
        h = min(1000, n_items*cfg.LINE_PLOT_HEIGHT)
        self.setFixedSize(w, h)
        self.addLayout(colspan=3)

        self.image_items = []
        self.plot_items = []
        for i in range(n_items):
            if show_image is True:
                vb = self.addViewBox(lockAspect=True, col=1)
                img = pg.ImageItem(np.random.normal(size=(100, 110)))
                vb.addItem(img)
                self.image_items.append(vb)
            self.plot_items.append(self.addPlot(col=2, colspan=2))
            self.nextRow()


class ImageViewWidget(pg.GraphicsLayoutWidget):
    def __init__(self, w, h, **kwargs):
        """Initialization."""
        super().__init__(**kwargs)

        self.setFixedSize(w, h)

        self._img = pg.ImageItem(border='w')
        self._view = self.addViewBox()
        self._view.addItem(self._img)

    def set_image(self, data):
        self._img.clear()
        self._img.setImage(data)
        self._view.autoRange()


class LinePlotWindow(QMainWindow):
    def __init__(self, n_items, *,
                 parent=None,
                 title="FXE instrument plot",
                 **kwargs):
        """Initialization."""
        super().__init__(parent=parent)

        self.setWindowTitle(title)
        self._cw = QWidget()
        self.setCentralWidget(self._cw)
        self._plot = LinePlotWidget(n_items, **kwargs)
        self.initUI()

        # scr_size = QDesktopWidget().screenGeometry()
        # x0 = int((scr_size.width() - self.frameSize().width()) / 2
        #          + random.randint(-100, 100))
        # y0 = int((scr_size.height() - self.frameSize().height()) / 2
        #          + random.randint(-100, 100))
        # self.move(x0, y0)

        # For real time plot
        # self.is_running = False
        # self.timer = pg.QtCore.QTimer()
        # self.timer.timeout.connect(self.update)
        # self.timer.start(200)

    def initUI(self):
        layout = QGridLayout()
        layout.addWidget(self._plot, 0, 0)

        self._cw.setLayout(layout)

    def update(self):
        pass
