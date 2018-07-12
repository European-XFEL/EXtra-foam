from math import sqrt
import sys
import time
import threading
import numpy as np

from karabo_bridge import Client
from silx.gui import qt

from data_processing import process_data
from plots import get_figures
import config as cfg
from logger import log


class UpdateThread(threading.Thread, qt.QMainWindow):
    def __init__(self, bridge_client):
        """
        :param client: A KaraboBridge client.
        """
        super(UpdateThread, self).__init__()
        self._init_window()

        self.client = bridge_client
        self.running = False
        self.first_loop = True

    def _init_window(self):
        # Initialise the main window
        qt.QMainWindow.__init__(self)
        self.setWindowTitle("LPD Integration")

        widget = qt.QWidget(self)
        self.setCentralWidget(widget)

        layout = qt.QGridLayout()
        widget.setLayout(layout)

        figures, _ = get_figures(widget)
        plots_per_line = int(sqrt(len(figures)))
        for k in range(len(figures)):
            try:
                name, plot  = figures[k]
                setattr(self, name, plot)
                layout.addWidget(plot, int(k / plots_per_line),
                                 k % plots_per_line)
            except IndexError:
                break

    def update_figures(self, data):
        """plot results"""
        title = ("Azimuthal Integration over {} pulses {}"
                 "".format(len(data["intensity"]), data["tid"]))
        self.setWindowTitle(title)

        for i, intensity in enumerate(data["intensity"]):
            self.integ.addCurveThreadSafe(data["momentum"],
                                          intensity,
                                          legend=str(i),
                                          copy=False,
                                          resetzoom=self.first_loop)

            self.normalised.addCurveThreadSafe(data["momentum"],
                                               intensity/intensity.max(),
                                               legend=str(i),
                                               copy=False,
                                               resetzoom=self.first_loop)

        for idx in cfg.ON_PULSES:
            p = getattr(self, "p{}".format(idx))
            ave = data["intensity"].mean(axis=0)
            # average over all pulses (in red)
            p.addCurveThreadSafe(data["momentum"], ave,
                                 legend="mean",
                                 copy=False,
                                 color="#FF0000")

            # the current pulse (in blue)
            p.addCurveThreadSafe(data["momentum"],
                                 data["intensity"][idx],
                                 legend="pulse", copy=False,
                                 resetzoom=False, color="#0000FF")

            # difference of the current pulse to the average over all pulses
            # (in bright gold)
            p.addCurveThreadSafe(data["momentum"],
                                 data["intensity"][idx] - ave,
                                 legend="diff",
                                 copy=False,
                                 color="#FDD017")

        # plot the image from the detector
        title = "Current train: {}".format(data["tid"])
        title += "\nPulse previewed: {}".format(cfg.VIEW_PULSE)
        self.plt2d.setGraphTitle(title)
        image = self._clip_image(data["images"][cfg.VIEW_PULSE])
        self.plt2d.addImage(image, replace=True, copy=False, yInverted=True)

    @staticmethod
    def _clip_image(array, min=-10000, max=800):
        x = array.copy()
        finite = np.isfinite(x)
        # Suppress warnings comparing numbers to nan
        with np.errstate(invalid='ignore'):
            x[finite & (x < min)] = np.nan
            x[finite & (x > max)] = np.nan
        return x

    def run(self):
        """Method implementing thread loop that gets data,
           integrates, and plots
        """
        self.running = True
        while self.running:
            # retrieve
            t0 = time.perf_counter()
            kb_data = self.client.next()
            log.info("Time for retrieving data from the server: {:.1f} ms"
                     .format(1000*(time.perf_counter() - t0)))

            # process
            t0 = time.perf_counter()
            data = process_data(kb_data)
            log.info("Time for processing the data: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))
            if data is None:
                continue

            # show
            t0 = time.perf_counter()
            self.update_figures(data)
            log.info("Time for updating the plots: {:.1f} ms"
                     .format(1000*(time.perf_counter() - t0)))

            self.first_loop = False

    def stop(self):
        """Stop the update thread"""
        self.running = False
        self.join()


if __name__ == '__main__':
    # addr = sys.argv[1] if len(sys.argv) > 1 else "tcp://10.253.0.53:4501"
    addr = sys.argv[1] if len(sys.argv) > 1 else "tcp://localhost:1236"
    client = Client(addr)

    global app
    app = qt.QApplication([])

    updateThread = UpdateThread(client)
    updateThread.setVisible(True)

    updateThread.start()
    app.exec_()
    updateThread.stop()
    sys.exit()
