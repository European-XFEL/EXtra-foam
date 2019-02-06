"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

PlotWidget

SinglePulseAiWidget
MultiPulseAiWidget
RoiIntensityMonitor

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .pyqtgraph import GraphicsView, intColor, mkPen, PlotItem, QtCore, QtGui
from .misc_widgets import PenFactory
from ..logger import logger
from ..config import config


class PlotWidget(GraphicsView):
    """GraphicsView widget displaying a single PlotItem.

    Note: it is different from the PlotWidget in pyqtgraph.

    This base class should be used to display plots except image in
    karaboFAI. For image, please refer to ImageView class.
    """
    # signals wrapped from PlotItem / ViewBox
    sigRangeChanged = QtCore.Signal(object, object)
    sigTransformChanged = QtCore.Signal(object)

    def __init__(self, parent=None, background='default', **kargs):
        """Initialization."""
        super().__init__(parent, background=background)
        if parent is not None:
            parent.registerPlotWidget(self)

        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                           QtGui.QSizePolicy.Expanding)
        self.enableMouse(False)
        self.plotItem = PlotItem(**kargs)
        self.setCentralItem(self.plotItem)

        self.plotItem.sigRangeChanged.connect(self.viewRangeChanged)

    def clear(self):
        """Remove all the items in the PlotItem object."""
        plot_item = self.plotItem
        for i in plot_item.items[:]:
            plot_item.removeItem(i)

    def reset(self):
        """Clear the data of all the items in the PlotItem object."""
        pass

    def update(self, data):
        raise NotImplemented

    def close(self):
        self.plotItem.close()
        self.plotItem = None
        self.setParent(None)
        super().close()

    def addItem(self, *args, **kwargs):
        """Explicitly call PlotItem.addItem.

        GraphicsView also has the addItem method.
        """
        self.plotItem.addItem(*args, **kwargs)

    def plot(self, *args, **kwargs):
        return self.plotItem.plot(*args, **kwargs)

    def setAspectLocked(self, *args, **kwargs):
        self.plotItem.setAspectLocked(*args, **kwargs)

    def setLabel(self, *args, **kwargs):
        self.plotItem.setLabel(*args, **kwargs)

    def setTitle(self, *args, **kwargs):
        self.plotItem.setTitle(*args, **kwargs)

    def addLegend(self, *args, **kwargs):
        self.plotItem.addLegend(*args, **kwargs)

    def hideAxis(self):
        for v in ["left", 'bottom']:
            self.plotItem.hideAxis(v)

    def viewRangeChanged(self, view, range):
        self.sigRangeChanged.emit(self, range)

    def saveState(self):
        return self.plotItem.saveState()

    def restoreState(self, state):
        return self.plotItem.restoreState(state)

    def closeEvent(self, QCloseEvent):
        parent = self.parent()
        if parent is not None:
            parent.unregisterPlotWidget(self)
        super().closeEvent(QCloseEvent)


class SinglePulseAiWidget(PlotWidget):
    """SinglePulseAiWidget class.

    A widget which allows user to visualize the the azimuthal integration
    result of individual pulses. The azimuthal integration result is also
    compared with the average azimuthal integration of all the pulses.
    """
    def __init__(self, *, pulse_id=0, plot_mean=True, parent=None):
        """Initialization.

        :param int pulse_id: ID of the pulse to be plotted.
        :param bool plot_mean: whether to plot the mean AI of all pulses
            if the data is pulse resolved.
        """
        super().__init__(parent=parent)

        self.pulse_id = pulse_id

        self.setLabel('left', "Scattering signal (arb. u.)")
        self.setLabel('bottom', "Momentum transfer (1/A)")

        self._pulse_plot = self.plot(name="pulse_plot", pen=PenFactory.yellow)

        if plot_mean:
            self._mean_plot = self.plot(name="mean", pen=PenFactory.cyan)
            self.addLegend(offset=(-40, 20))
        else:
            self._mean_plot = None

    def clear(self):
        """Override."""
        self.reset()

    def reset(self):
        """Override."""
        self._pulse_plot.setData([], [])
        if self._mean_plot is not None:
            self._mean_plot.setData([], [])

    def update(self, data):
        """Override."""
        momentum = data.momentum
        intensities = data.intensities

        if intensities is None:
            return

        if intensities.ndim == 2:
            # pulse resolved data
            max_id = len(data.images) - 1
            if self.pulse_id <= max_id:
                self._pulse_plot.setData(momentum,
                                         intensities[self.pulse_id])
            else:
                logger.error("<VIP pulse ID>: VIP pulse ID ({}) > Maximum "
                             "pulse ID ({})".format(self.pulse_id, max_id))
                return
        else:
            self._pulse_plot.setData(momentum, intensities)

        if self._mean_plot is not None:
            self._mean_plot.setData(momentum, data.intensity_mean)


class MultiPulseAiWidget(PlotWidget):
    """MultiPulseAiWidget class.

    Widget used for displaying azimuthal integration result for all
    the pulses in a train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._n_pulses = 0

        self.setLabel('bottom', "Momentum transfer (1/A)")
        self.setLabel('left', "Scattering signal (arb. u.)")

    def clear(self):
        """Override."""
        self.reset()

    def reset(self):
        """Override."""
        for item in self.plotItem.items:
            item.setData([], [])

    def update(self, data):
        """Override."""
        momentum = data.momentum
        intensities = data.intensities

        if intensities is None:
            return

        n_pulses = len(intensities)
        if n_pulses != self._n_pulses:
            self._n_pulses = n_pulses
            # re-plot if number of pulses change
            self.clear()
            for i, intensity in enumerate(intensities):
                self.plot(momentum, intensity,
                          pen=mkPen(intColor(i, hues=9, values=5), width=2))
        else:
            for item, intensity in zip(self.plotItem.items, intensities):
                item.setData(momentum, intensity)


class RoiIntensityMonitor(PlotWidget):
    """RoiIntensityMonitor class.

    Widget used for displaying the evolution of the integration of ROIs.
    """
    def __init__(self, *, window=600, parent=None):
        """Initialization.

        :param int window: window size, i.e. maximum number of trains to
            display. Default = 600.
        """
        super().__init__(parent=parent)

        self._window = window

        self.setLabel('bottom', "Train ID")
        self.setLabel('left', "Intensity (arb. u.)")
        self.addLegend(offset=(-40, 20))

        self._roi1_plot = self.plot(
            name="ROI 1", pen=PenFactory.__dict__[config["ROI_COLORS"][0]])
        self._roi2_plot = self.plot(
            name="ROI 2", pen=PenFactory.__dict__[config["ROI_COLORS"][1]])

    def clear(self):
        """Override."""
        self.reset()

    def reset(self):
        """Override."""
        self._roi1_plot.setData([], [])
        self._roi2_plot.setData([], [])

    def update(self, data):
        """Override."""
        train_ids = data.roi_train_ids[-self._window:]
        roi1_intensities = data.roi1_intensities[-self._window:]
        roi2_intensities = data.roi2_intensities[-self._window:]

        self._roi1_plot.setData(train_ids, roi1_intensities)
        self._roi2_plot.setData(train_ids, roi2_intensities)

    @QtCore.pyqtSlot(int)
    def onWindowSizeChanged(self, v):
        self._window = v
