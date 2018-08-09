"""
Offline and online data analysis tool for Azimuthal integration at
FXE instrument, European XFEL.

Plot widgets module.

Author: Jun Zhu, jun.zhu@xfel.eu, zhujun981661@gmail.com
Copyright (C) European XFEL GmbH Hamburg. All rights reserved.
"""
import numpy as np

from .pyqtgraph.Qt import QtGui
from .pyqtgraph import GraphicsLayoutWidget, ImageItem, mkPen
from .config import Config as cfg


class MainLinePlotWidget(GraphicsLayoutWidget):
    def __init__(self, parent=None, **kwargs):
        """Initialization.

        :param int n_lines: Number of PlotItems.
        """
        super().__init__(parent, **kwargs)

        w = cfg.MAIN_WINDOW_WIDTH - cfg.MAIN_LINE_PLOT_HEIGHT - 25
        h = cfg.MAIN_LINE_PLOT_HEIGHT
        self.setFixedSize(w, h)

        self._plot = self.addPlot()
        self._plot.setTitle("")
        self._plot.setLabel('bottom', cfg.X_LABEL)
        self._plot.setLabel('left', cfg.Y_LABEL)

    def set_title(self, text):
        self._plot.setTitle(text)

    def clear_(self):
        self._plot.clear()

    def update(self, *args, **kwargs):
        self._plot.plot(*args, **kwargs)


class ImageViewWidget(GraphicsLayoutWidget):
    def __init__(self, parent=None, **kwargs):
        """Initialization."""
        super().__init__(parent, **kwargs)

        self.setFixedSize(cfg.MAIN_LINE_PLOT_HEIGHT, cfg.MAIN_LINE_PLOT_HEIGHT)

        self._img = ImageItem(border='w')
        self._view = self.addViewBox(lockAspect=True)
        self._view.addItem(self._img)

    def clear_(self):
        self._img.clear()

    def update(self, *args, **kwargs):
        self._img.setImage(*args, **kwargs)
        self._view.autoRange()


class LinePlotWindow(QtGui.QMainWindow):
    def __init__(self, window_id, pulse_ids, *,
                 parent=None,
                 show_image=False,
                 title="FXE Azimuthal integration",
                 **kwargs):
        """Initialization."""
        super().__init__(parent=parent)

        self._id = window_id
        self._pulse_ids = pulse_ids
        self.setWindowTitle(title)
        self._cw = QtGui.QWidget()
        self.setCentralWidget(self._cw)

        self._show_image = show_image
        self.image_items = []
        self.plot_items = []

        self.initUI()

    def initUI(self):
        gl_widget = GraphicsLayoutWidget()
        g_layout = gl_widget.ci.layout
        g_layout.setColumnStretchFactor(0, 1)
        g_layout.setColumnStretchFactor(1, 1)
        g_layout.setColumnStretchFactor(2, 1)

        w = cfg.LINE_PLOT_WIDTH + self._show_image*(cfg.LINE_PLOT_HEIGHT - 20)
        h = min(1000, len(self._pulse_ids)*cfg.LINE_PLOT_HEIGHT)
        gl_widget.setFixedSize(w, h)

        for pulse_id in self._pulse_ids:
            gl_widget.addLabel("Pulse No. {:04d}".format(pulse_id),
                               colspan=3)
            gl_widget.nextRow()

            if self._show_image is True:
                vb = gl_widget.addViewBox(lockAspect=True, colspan=1)
                img = ImageItem(border='w')
                vb.addItem(img)
                self.image_items.append(img)

                p = gl_widget.addPlot(col=1, colspan=2)
            else:
                p = gl_widget.addPlot(col=0, colspan=3)

            if pulse_id == self._pulse_ids[-1]:
                # all plots share one x label
                p.setLabel('bottom', cfg.X_LABEL)
            else:
                p.setLabel('bottom', '')
            p.setLabel('left', cfg.Y_LABEL)
            self.plot_items.append(p)
            gl_widget.nextRow()

        layout = QtGui.QGridLayout()
        layout.addWidget(gl_widget)
        self._cw.setLayout(layout)

    def update(self, data):
        for i, pulse_id in enumerate(self._pulse_ids):
            p = self.plot_items[i]
            if data is not None:
                p.plot(data["momentum"], data["intensity"][pulse_id],
                       name="origin",
                       pen=mkPen(cfg.CUSTOM_PEN[0]['color'],
                                 width=cfg.CUSTOM_PEN[0]['width']))

                ave = np.mean(data["intensity"], axis=0)

                p.plot(data["momentum"], ave,
                       name="mean",
                       pen=mkPen(cfg.CUSTOM_PEN[1]['color'],
                                 width=cfg.CUSTOM_PEN[1]['width']))

                p.plot(data["momentum"], data["intensity"][pulse_id] - ave,
                       name="difference",
                       pen=mkPen(cfg.CUSTOM_PEN[2]['color'],
                                 width=cfg.CUSTOM_PEN[2]['width']))

                if i == 0:
                    p.addLegend(offset=cfg.LINE_PLOT_LEGEND_OFFSET)

            if data is not None and self._show_image is True:
                self.image_items[i].setImage(data["image"][pulse_id])

    def clear(self):
        for item in self.plot_items:
            item.clear()
        for item in self.image_items:
            item.clear()

    def closeEvent(self, QCloseEvent):
        super().closeEvent(QCloseEvent)
        self.parent().remove_window(self._id)
