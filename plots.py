from collections import deque

from silx.gui.plot import Plot1D, Plot2D, PlotWidget, PlotWindow
from silx.gui.plot.utils.axis import SyncAxes
from silx.gui import qt

import config as cfg


class PlotTrendLine(PlotWidget):

    def __init__(self, size=10, parent=None, backend=None):
        super(PlotTrendLine, self).__init__(parent=parent, backend=backend,
                                            resetzoom=True, autoScale=True,
                                            logScale=True, grid=True,
                                            curveStyle=True, colormap=False,
                                            aspectRatio=False, yInverted=False,
                                            copy=True, save=True, print_=True,
                                            control=True, position=True,
                                            roi=True, mask=False, fit=True)
        self._buffer = deque(maxlen=size)

        if parent is None:
           self.setWindowTitle('PlotTrendLine')
        self.getXAxis().setLabel('X')
        self.getYAxis().setLabel('Y')

    def addPoint(self, x, **kw):
       self._buffer.append(x)
       x_axis = [i for i in range(len(self._buffer))]
       self.addCurve(x_axis, list(self._buffer), **kw)


class ThreadSafePlotTrendLine(PlotTrendLine):
    _sigAddPoint = qt.Signal(tuple, dict)

    def __init__(self, size=10, parent=None, backend=None):
        super(ThreadSafePlotTrendLine, self).__init__(size, parent, backend)
        self._sigAddPoint.connect(self.__addPoint)

    def __addPoint(self, args, kwargs):
        self.addPoint(*args, **kwargs)

    def addPointThreadSafe(self, *args, **kwargs):
        self._sigAddPoint.emit(args, kwargs)


class ThreadSafePlotWidget(PlotWidget):
    _sigAddCurve = qt.Signal(tuple, dict)

    def __init__(self, parent=None, backend=None):
        super(ThreadSafePlotWidget, self).__init__(parent, backend)
        self._sigAddCurve.connect(self.__addCurve)

    def __addCurve(self, args, kwargs):
        self.addCurve(*args, **kwargs)

    def addCurveThreadSafe(self, *args, **kwargs):
        self._sigAddCurve.emit(args, kwargs)


class ThreadSafePlot1D(Plot1D):
    _sigAddCurve = qt.Signal(tuple, dict)

    def __init__(self, parent=None, backend=None):
        super(ThreadSafePlot1D, self).__init__(parent, backend)
        self._sigAddCurve.connect(self.__addCurve)

    def __addCurve(self, args, kwargs):
        self.addCurve(*args, **kwargs)

    def addCurveThreadSafe(self, *args, **kwargs):
        self._sigAddCurve.emit(args, kwargs)


class ThreadSafePlot2D(Plot2D):
    _sigAddCurve = qt.Signal(tuple, dict)

    def __init__(self, parent=None, backend=None):
        super(ThreadSafePlot2D, self).__init__(parent, backend)
        self._sigAddCurve.connect(self.__addCurve)

    def __addCurve(self, args, kwargs):
        self.addCurve(*args, **kwargs)

    def addCurveThreadSafe(self, *args, **kwargs):
        self._sigAddCurve.emit(args, kwargs)


def get_figures(widget):
    figures = []

    plot2d = ThreadSafePlot2D(parent=widget)
    plot2d.getDefaultColormap().setName('viridis')
    plot2d.getDefaultColormap().setVMin(-10)
    plot2d.getDefaultColormap().setVMax(6000)
    plot2d.setGraphYLabel("")
    plot2d.setGraphXLabel("")
    figures.append(('plt2d', plot2d))

    integ = ThreadSafePlotWidget(parent=widget)
    integ.setGraphYLabel("Scattering Signal, S(q) [arb. u.]")
    integ.setGraphXLabel("Momentum Transfer, q[1/A]")
    integ.setGraphTitle("Integration")
    figures.append(('integ', integ))

    normalised = ThreadSafePlotWidget(parent=widget)
    normalised.setGraphYLabel("Normalised Scattering Signal, S(q) [arb. u.]")
    normalised.setGraphXLabel("Momentum Transfer, q[1/A]")
    normalised.setGraphTitle("Normalised")
    figures.append(('normalised', normalised))

    for idx in cfg.ON_PULSES:
        p = ThreadSafePlotWidget(parent=widget)
        p.setGraphYLabel("Normalised Scattering Signal, S(q) [arb. u.]")
        p.setGraphXLabel("Momentum Transfer, q[1/A]")
        p.setGraphTitle("Pulse {}".format(idx))
        figures.append(("p{}".format(idx), p))

        integ = ThreadSafePlotTrendLine(parent=widget)
        integ.setGraphYLabel("Integrated Mean of pulse difference")
        integ.setGraphXLabel("Last {} pulses".format(integ._buffer.maxlen))
        integ.setGraphTitle("Pulse {}".format(idx))
        figures.append(("pinteg{}".format(idx), integ))

    constraints = [] #[SyncAxes([p.getXAxis() for _, p in figures]),
                  # SyncAxes([p.getYAxis() for _, p in figures])]

    return figures, constraints
