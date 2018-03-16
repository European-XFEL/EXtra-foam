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

import threading
import time
import random
import sys
from silx.gui import qt
from silx.gui.plot import Plot1D, Plot2D
from karabo_bridge import KaraboBridge
from lpd_tools import LPDConfiguration, offset_image
from integration import integrate

config = LPDConfiguration(hole_size=26.28e-3, q_offset=3)


class ThreadSafePlot1D(Plot1D):
    """Add a thread-safe :meth:`addCurveThreadSafe` method to Plot1D.
    """
    _sigAddCurve = qt.Signal(tuple, dict)
    """Signal used to perform addCurve in the main thread.
        It takes args and kwargs as arguments.
    """

    def __init__(self, parent=None):
        super(ThreadSafePlot1D, self).__init__(parent)
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
    :param client: A KaraboBridge client.
    :param plot1d: The ThreadSafePlot1D to update.
    :param plot2d: The ThreadSafePlot2D to update.
    """

    def __init__(self, client, plot1d, plot2d):
        self.client = client
        self.plot1d = plot1d
        self.plot2d = plot2d
        self.running = False
        super(UpdateThread, self).__init__()

    def run(self):
        """Method implementing thread loop that updates the plot"""
        self.running = True
        while self.running:
            # retreive
            t_start = time.time()
            data = self.client.next()
            images = data.pop("FXE_DET_LPD1M-1/DET/combined")["image.data"]
            tid = data.popitem()[1]["detector.trainId"]
            retrieval_t = time.time() - t_start

            # integrate
            t_start = time.time()
            integ_result = integrate(images)
            integ_t = time.time() - t_start

            # plot the result of the integration
            t_start = time.time()
            title = ("Azimuthal Integration over {} pulses {}"
                     "".format(len(integ_result), tid))
            self.plot1d.setGraphTitle(title)

            allTheMomentums = []

            for index in range(len(integ_result)):
                entry = integ_result[index]
                scattering = [item for items in entry[0] for item in items]
                momentum = [item for items in entry[1] for item in items]
                allTheMomentums.append(momentum)
                self.plot1d.addCurveThreadSafe(scattering, momentum,
                                               legend=str(index),
                                               copy=False)

            # plot the image from the detector
            pulse_id = random.randint(0, len(images)-1)
            title = "Current train: {}".format(tid)
            title += "\nPulse previewed: {}".format(pulse_id)
            self.plot2d.setGraphTitle(title)
            image = offset_image(config, images[pulse_id])
            self.plot2d.addImage(image, replace=True,
                                 copy=False, yInverted=True)
            plot_t = time.time() - t_start
            print(tid, "retreival", retrieval_t,
                  "integration", integ_t, "plot", plot_t)

    def stop(self):
        """Stop the update thread"""
        self.running = False
        self.join(2)


def main():
    client = KaraboBridge("tcp://localhost:4545")
    global app
    app = qt.QApplication([])

    plot1d = ThreadSafePlot1D()
    plot1d.setGraphYLabel("Scattering Signal, S(q) [arb. u.]")
    plot1d.setGraphXLabel("Momentum Transfer, q[1/A]")
    plot1d.show()

    plot2d = ThreadSafePlot2D()
    plot2d.getDefaultColormap().setName('viridis')
    plot2d.getDefaultColormap().setVMin(-10)
    plot2d.getDefaultColormap().setVMax(6000)
    plot2d.setGraphYLabel("")
    plot2d.setGraphXLabel("")
    plot2d.addImage
    plot2d.show()

    # Create the thread that gets the data from the bridge
    # and calls ThreadSafePlotNd.addCurveThreadSafe
    updateThread = UpdateThread(client, plot1d, plot2d)
    updateThread.start()  # Start updating the plot

    try:
        app.exec_()
    except KeyboardInterrupt:
        updateThread.stop()  # Stop updating the plot
        sys.exit()


if __name__ == '__main__':
    main()
