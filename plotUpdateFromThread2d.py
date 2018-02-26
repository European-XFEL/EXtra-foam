# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
"""This script illustrates the update of a :mod:`silx.gui.plot` widget from a
thread.

The problem is that plot and GUI methods should be called from the main thread.
To safely update the plot from another thread, one need to make the update
asynchronously from the main thread.
In this example, this is achieved through a Qt signal.

In this example we create a subclass of
:class:`~silx.gui.plot.PlotWindow.Plot2D`
that adds a thread-safe method to add curves:
:meth:`ThreadSafePlot1D.addCurveThreadSafe`.
This thread-safe method is then called from a thread to update the plot.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "05/09/2017"


import threading

from silx.gui import qt
from silx.gui.plot import Plot2D

from karabo_bridge import KaraboBridge
from lpd_tools import LPDConfiguration, offset_image


config = LPDConfiguration(hole_size=-26.28e-3, q_offset=3)
client = KaraboBridge("tcp://localhost:4545")

Nx = 150
Ny = 50


class ThreadSafePlot2D(Plot2D):
    """Add a thread-safe :meth:`addCurveThreadSafe` method to Plot2D.
    """

    _sigAddCurve = qt.Signal(tuple, dict)
    """Signal used to perform addCurve in the main thread.

    It takes args and kwargs as arguments.
    """

    def __init__(self, parent=None):
        super(ThreadSafePlot2D, self).__init__(parent)
        # Connect the signal to the method actually calling addCurve
        self._sigAddCurve.connect(self.__addCurve)

    def __addCurve(self, args, kwargs):
        """Private method calling addCurve from _sigAddCurve"""
        self.addCurve(*args, **kwargs)

    def addCurveThreadSafe(self, *args, **kwargs):
        """Thread-safe version of :meth:`silx.gui.plot.Plot.addCurve`

        This method takes the same arguments as Plot.addCurve.

        WARNING: This method does not return a value as opposed to
                 Plot.addCurve
        """
        self._sigAddCurve.emit(args, kwargs)


class UpdateThread(threading.Thread):
    """Thread updating the curve of a :class:`ThreadSafePlot2D`

    :param plot2d: The ThreadSafePlot2D to update."""

    def __init__(self, plot2d):
        self.plot2d = plot2d
        self.running = False
        super(UpdateThread, self).__init__()

    def start(self):
        """Start the update thread"""
        self.running = True
        super(UpdateThread, self).start()

    def run(self, pos={'x0': 0, 'y0': 0}):
        """Method implementing thread loop that updates the plot"""
        while self.running:
            data = client.next()
            d = data["FXE_DET_LPD1M-1/DET/combined"]["image.data"]
            image = offset_image(config, d[0])
            # plot the data
            self.plot2d.addImage(image, replace=True)

    def stop(self):
        """Stop the update thread"""
        self.running = False
        self.join(2)


def main():
    global app
    app = qt.QApplication([])

    # Create a ThreadSafePlot2D, set its limits and display it
    plot2d = ThreadSafePlot2D()
    plot2d.setLimits(-10, 6000, Nx, Ny)
    plot2d.getDefaultColormap().setName('viridis')
    plot2d.getDefaultColormap().setVMin(-10)
    plot2d.getDefaultColormap().setVMax(6000)
    plot2d.show()

    # Create the thread that calls ThreadSafePlot2D.addCurveThreadSafe
    updateThread = UpdateThread(plot2d)
    updateThread.start()  # Start updating the plot

    app.exec_()

    updateThread.stop()  # Stop updating the plot


if __name__ == '__main__':
    main()
