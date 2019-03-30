"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Base PlotWidget and various concrete PlotWidgets.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .. import pyqtgraph as pg

from ..pyqtgraph import GraphicsView, PlotItem, QtCore, QtGui

from ..misc_widgets import make_brush, make_pen
from ...logger import logger
from ...config import config


class PlotWidget(GraphicsView):
    """GraphicsView widget displaying a single PlotItem.

    Note: it is different from the PlotWidget in pyqtgraph.

    This base class should be used to display plots except image in
    karaboFAI. For image, please refer to ImageView class.
    """
    class BarGraphItem(pg.BarGraphItem):
        def setData(self, x, height):
            """PlotItem interface."""
            self.setOpts(x=x, height=height)

    class ErrorBarItem(pg.GraphicsObject):
        """ErrorBarItem.

        This is a re-implementation of pg.ErrorBarItem. It is supposed
        to be much faster.
        """
        def __init__(self, x=None, y=None, y_min=None, y_max=None, beam=None,
                     pen=None):
            """Initialization.

            Note: y is not used for now.
            """
            super().__init__()

            self._path = None

            self._x = None
            self._y = None
            self._y_min = None
            self._y_max = None

            self._beam = 0.0 if beam is None else beam
            self._pen = make_pen('e') if pen is None else pen

            self.setData(x, y, y_min=y_min, y_max=y_max)

        def setData(self, x, y, y_min=None, y_max=None, beam=None, pen=None):
            """PlotItem interface."""
            self._x = [] if x is None else x
            self._y = [] if y is None else y

            self._y_min = self._y if y_min is None else y_min
            self._y_max = self._y if y_max is None else y_max

            if len(self._x) != len(self._y):
                raise ValueError("'x' and 'y' data have different lengths!")
            if not len(self._y) == len(self._y_min) == len(self._y_max):
                raise ValueError(
                    "'y_min' and 'y_max' data have different lengths!")

            if beam is not None and beam >= 0.0:
                self._beam = beam
            if pen is not None:
                self._pen = pen

            self._path = None
            self.update()
            self.prepareGeometryChange()
            self.informViewBoundsChanged()

        def drawPath(self):
            p = QtGui.QPainterPath()

            x = self._x

            for i in range(len(x)):
                # plot the lower horizontal lines
                p.moveTo(x[i] - self._beam / 2., self._y_min[i])
                p.lineTo(x[i] + self._beam / 2., self._y_min[i])

                # plot the vertical line
                p.moveTo(x[i], self._y_min[i])
                p.lineTo(x[i], self._y_max[i])

                # plot the upper horizontal line
                p.moveTo(x[i] - self._beam / 2., self._y_max[i])
                p.lineTo(x[i] + self._beam / 2., self._y_max[i])

            self._path = p
            self.prepareGeometryChange()

        def paint(self, p, *args):
            if self._path is None:
                self.drawPath()

            p.setPen(self._pen)
            p.drawPath(self._path)

        def boundingRect(self):
            if self._path is None:
                self.drawPath()
            return self._path.boundingRect()

    # signals wrapped from PlotItem / ViewBox
    sigRangeChanged = QtCore.Signal(object, object)
    sigTransformChanged = QtCore.Signal(object)

    _pen = make_pen(None)
    _brush_size = 10

    def __init__(self, parent=None, background='default', **kargs):
        """Initialization."""
        super().__init__(parent, background=background)
        if parent is not None:
            parent.registerPlotWidget(self)

        self._data = None  # keep the last data (could be invalid)

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
        for item in self.plotItem.items:
            item.setData([], [])

    def update(self, data):
        raise NotImplemented

    def close(self):
        self.plotItem.close()
        self.plotItem = None
        self.setParent(None)
        super().close()

    def addItem(self, *args, **kwargs):
        """Explicitly call PlotItem.addItem.

        This method must be here to override the addItem method in
        GraphicsView. Otherwise, people may misuse the addItem method.
        """
        self.plotItem.addItem(*args, **kwargs)

    def removeItem(self, *args, **kwargs):
        self.plotItem.removeItem(*args, **kwargs)

    def plotCurve(self, *args, **kwargs):
        """Add and return a new curve plot."""
        item = pg.PlotCurveItem(*args, **kwargs)
        self.plotItem.addItem(item)
        return item

    def plotScatter(self, *args, **kwargs):
        """Add and return a new scatter plot."""
        item = pg.ScatterPlotItem(*args,
                                  pen=self._pen,
                                  size=self._brush_size, **kwargs)
        self.plotItem.addItem(item)
        return item

    def plotBar(self, x=None, height=None, width=1.0, **kwargs):
        """Add and return a new bar plot."""
        if x is None:
            x = []
            height = []
        item = self.BarGraphItem(x=x, height=height, width=width, **kwargs)
        self.plotItem.addItem(item)
        return item

    def plotErrorBar(self, x=None, y=None, y_min=None, y_max=None, beam=None):
        item = self.ErrorBarItem(x=x, y=y, y_min=y_min, y_max=y_max, beam=beam)
        self.plotItem.addItem(item)
        return item

    def plotImage(self, *args, **kargs):
        """Add and return a image item."""
        # TODO: this will be done when another branch is merged
        raise NotImplemented

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

    def showAxis(self):
        for v in ["left", 'bottom']:
            self.plotItem.showAxis(v)

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
    result of a single pulse. The azimuthal integration result is also
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

        if plot_mean:
            self.addLegend(offset=(-40, 20))

        self._pulse_plot = self.plotCurve(name="pulse_plot", pen=make_pen("y"))

        if plot_mean:
            self._mean_plot = self.plotCurve(name="mean", pen=make_pen("b"))
        else:
            self._mean_plot = None

    def update(self, data):
        """Override."""
        momentum = data.momentum
        intensities = data.intensities

        if intensities is None:
            return

        if intensities.ndim == 2:
            # pulse resolved data
            max_id = data.n_pulses - 1
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

    Widget for displaying azimuthal integration result for all
    the pulses in a train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._n_pulses = 0

        self.setLabel('bottom', "Momentum transfer (1/A)")
        self.setLabel('left', "Scattering signal (arb. u.)")
        self.setTitle(' ')

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
                self.plotCurve(momentum, intensity,
                               pen=make_pen(i, hues=9, values=5))
        else:
            for item, intensity in zip(self.plotItem.items, intensities):
                item.setData(momentum, intensity)


class SampleDegradationWidget(PlotWidget):
    """SampleDegradationWindow class.

    A widget which allows users to monitor the degradation of the sample
    within a train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._plot = self.plotBar(width=0.6, brush='b')
        self.addItem(self._plot)

        self.setLabel('left', "Integrated difference (arb.)")
        self.setLabel('bottom', "Pulse ID")
        self.setTitle('FOM with respect to the first pulse')

    def update(self, data):
        """Override."""
        foms = data.sample_degradation_foms
        if foms is None:
            return

        self._plot.setData(range(len(foms)), foms)


class RoiValueMonitor(PlotWidget):
    """RoiValueMonitor class.

    Widget for displaying the evolution of the value (integration, median,
    mean) of ROIs.
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
        self.setTitle(' ')
        self.addLegend(offset=(-40, 20))

        self._plots = []
        for i, c in enumerate(config["ROI_COLORS"], 1):
            self._plots.append(self.plotCurve(name=f"ROI {i}", pen=make_pen(c)))

    def update(self, data):
        """Override."""
        for i, plot in enumerate(self._plots, 1):
            tids, roi_hist, _ = getattr(data.roi, f"roi{i}_hist")
            plot.setData(tids[-self._window:], roi_hist[-self._window:])

    @QtCore.pyqtSlot(int)
    def onDisplayRangeChange(self, v):
        self._window = v


class CorrelationWidget(PlotWidget):
    """CorrelationWidget class.

    Widget for displaying correlations between FOM and different parameters.
    """
    _colors = ['g', 'b', 'y', 'p']
    _brushes = {
        0: make_brush(_colors[0], 120),
        1: make_brush(_colors[1], 120),
        2: make_brush(_colors[2], 120),
        3: make_brush(_colors[3], 120)
    }
    _opaque_brushes = {
        0: make_brush(_colors[0]),
        1: make_brush(_colors[1]),
        2: make_brush(_colors[2]),
        3: make_brush(_colors[3])
    }

    def __init__(self, idx, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._idx = idx

        self.setLabel('left', "FOM (arb. u.)")
        self.setLabel('bottom', "Correlator (arb. u.)")
        self.setTitle(' ')

        self._bar = self.plotErrorBar()
        self._plot = self.plotScatter(brush=self._brushes[self._idx])

        self._device_id = None
        self._ppt = None
        self._resolution = 0.0

    def update(self, data):
        """Override."""
        try:
            correlator, foms, info = getattr(data.correlation,
                                             f'param{self._idx}')
        except AttributeError:
            return

        device_id = info['device_id']
        ppt = info['property']
        if self._device_id != device_id or self._ppt != ppt:
            self.setLabel('bottom', f"{device_id + ' | ' + ppt} (arb. u.)")
            self._device_id = device_id
            self._ppt = ppt

        if isinstance(foms, list):
            if self._resolution != 0.0:
                self._resolution = 0.0
                self._bar.setData([], [], beam=0.0)
                self._plot.setBrush(self._brushes[self._idx])

            self._plot.setData(correlator, foms)
            # make auto-range of the viewbox work correctly
            self._bar.setData(correlator[:1], foms[:1])
        else:
            resolution = info['resolution']

            if self._resolution != resolution:
                self._resolution = resolution
                self._bar.setData([], [], beam=resolution)
                self._plot.setBrush(self._opaque_brushes[self._idx])

            self._bar.setData(x=correlator,
                              y=foms.avg, y_min=foms.min, y_max=foms.max)
            self._plot.setData(correlator, foms.avg)


class LaserOnOffFomWidget(PlotWidget):
    """LaserOnOffFomWidget class.

    Widget for displaying the evolution of FOM in the Laser On-off analysis.
    """

    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('bottom', "Train ID")
        self.setLabel('left', "ROI (arb. u.)")
        self.setTitle(' ')

        self._plot = self.plotScatter(brush=make_brush('o'))

    def update(self, data):
        """Override."""
        tids, foms, _ = data.on_off.foms
        self._plot.setData(tids, foms)


class LaserOnOffAiWidget(PlotWidget):
    """LaserOnOffAiWidget class.

    Widget for displaying the average of the azimuthal integrations
    of laser-on/off pulses.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('left', "Scattering signal (arb. u.)")
        self.setLabel('bottom', "Momentum transfer (1/A)")
        self.setTitle('Moving average of on- and off- pulses')
        self.addLegend(offset=(-60, 20))

        self._on_pulse = self.plotCurve(name="Laser-on", pen=make_pen("p"))
        self._off_pulse = self.plotCurve(name="Laser-off", pen=make_pen("g"))

    def update(self, data):
        """Override."""
        momentum = data.momentum
        on_pulse = data.on_off.on_pulse
        off_pulse = data.on_off.off_pulse

        if on_pulse is None:
            self._data = None
        else:
            if off_pulse is None:
                if self._data is None:
                    return
                # on-pulse arrives but off-pulse does not
                momentum, on_pulse, off_pulse = self._data
            else:
                self._data = (momentum, on_pulse, off_pulse)

            self._on_pulse.setData(momentum, on_pulse)
            self._off_pulse.setData(momentum, off_pulse)


class LaserOnOffDiffWidget(PlotWidget):
    """LaserOnOffDiffWidget class.

    Widget for displaying the difference of the average of the azimuthal
    integrations of laser-on/off pulses.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('left', "Scattering signal (arb. u.)")
        self.setLabel('bottom', "Momentum transfer (1/A)")
        self.setTitle('Moving average of on-off')

        self._plot = self.plotCurve(name="On - Off", pen=make_pen("y"))

    def clear(self):
        """Override."""
        self.reset()

    def reset(self):
        """Override."""
        self._plot.setData([], [])

    def update(self, data):
        """Override."""
        momentum = data.momentum
        on_pulse = data.on_off.on_pulse
        off_pulse = data.on_off.off_pulse
        diff = data.on_off.diff

        if on_pulse is None:
            self._data = None
        else:
            if off_pulse is None:
                if self._data is None:
                    return
                # on-pulse arrives but off-pulse does not
                diff = self._data
            else:
                self._data = diff

            self._plot.setData(momentum, diff)


class XasSpectrumWidget(PlotWidget):
    """XasSpectrumWidget class.

    Widget for displaying the XAS spectrum.
    """

    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('bottom', "Energy (eV)")
        self.setLabel('left', "Absorption")
        self.setTitle(' ')

        self._plot = self.plotScatter(brush=make_brush('b'))

    def update(self, data):
        """Override."""
        pass


class XasSpectrumDiffWidget(PlotWidget):
    """XasSpectrumDiffWidget class.

    Widget for displaying the difference of two XAS spectra.
    """

    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('bottom', "Energy (eV)")
        self.setLabel('left', "Absorption")
        self.setTitle(' ')

        self._plot = self.plotScatter(brush=make_brush('b'))

    def update(self, data):
        """Override."""
        pass
