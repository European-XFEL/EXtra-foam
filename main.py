import random
import sys
import threading
import time

from karabo_bridge import Client
from silx.gui import qt

from constants import bp_map_file, bridge_key, running_averages
from integration import integrate, running_mean, differences, diff_integrals
from plots import get_figures


class UpdateThread(threading.Thread, qt.QMainWindow):
    def __init__(self, bridge_client):
        """
        :param client: A KaraboBridge client.
        """
        super(UpdateThread, self).__init__()
        self.__init_window()

        self.client = bridge_client
        self.running = False

    def __init_window(self):
        # Initialise the main window
        qt.QMainWindow.__init__(self)
        self.setWindowTitle("LPD Integration")

        widget = qt.QWidget(self)
        self.setCentralWidget(widget)

        layout = qt.QGridLayout()
        widget.setLayout(layout)

        figures, _ = get_figures(widget)
        k = 0
        plots_per_line = 4
        for k in range(len(figures)):
            try:
                name, plot  = figures[k]
                setattr(self, name, plot)
                layout.addWidget(plot, int(k / plots_per_line),
                                 k % plots_per_line)
            except IndexError:
                break

    def update_figures(self, momentum, azi, normalised, means, 
                       diffs, diffs_integs, images, tid):
            # plot results
            title = ("Azimuthal Integration over {} pulses {}"
                     "".format(len(azi), tid))
            self.setWindowTitle(title)

            assert len(azi) == len(normalised)

            for index in range(len(azi)):
                scattering = azi[index]
                norm_scattering = normalised[index]

                self.integ.addCurveThreadSafe(momentum, scattering,
                                               legend=str(index),
                                               copy=False,
                                               resetzoom=False)

                self.normalised.addCurveThreadSafe(momentum, norm_scattering,
                                                   legend=str(index),
                                                   copy=False,
                                                   resetzoom=False)

            # Red is the difference between the running average and the
            # current pulse, pink is the running mean, and blue the current
            # pulse
            for idx in running_averages:
                p = getattr(self, "p{}".format(idx))
                p.addCurveThreadSafe(momentum,
                                     means[idx]['scattering'],
                                     legend="mean",copy=False,
                                     color="#FF0000")
                p.addCurveThreadSafe(momentum, diffs[idx],
                                     legend="diff",copy=False,
                                     color="#FF9099")
                p.addCurveThreadSafe(momentum, normalised[idx],
                                     legend="pulse", copy=False,
                                     resetzoom=False, color="#0000FF")

                p = getattr(self, "pinteg{}".format(idx))
                p.addPointThreadSafe(diffs_integs[idx])


            # plot the image from the detector
            pulse_id = random.randint(0, len(azi)-1)
            title = "Current train: {}".format(tid)
            title += "\nPulse previewed: {}".format(pulse_id)
            self.plt2d.setGraphTitle(title)
            image = images[..., pulse_id]
            self.plt2d.addImage(image, replace=True,
                                 copy=False, yInverted=True)

    def run(self):
        """Method implementing thread loop that gets data, 
           integrates, and plots
        """
        bp_map = None
        if bp_map_file:
            import h5py
            f = h5py.File(bp_map_file, "r")
            bp_map = f["/MappedBadPixels"][()][::-1,::-1,:]
            f.close()
            print("Using bad pixel mask of shape {}".format(bp_map.shape))

        self.running = True

        while self.running:
            # retrieve
            t_start = time.time()
            data = self.client.next()
            images = data[bridge_key]["image.data"]
            cells = data[bridge_key]["image.cellId"]
            tid = data[bridge_key]["header.trainId"]
            retrieval_t = time.time() - t_start

            if bp_map is not None:
                import copy
                images = copy.copy(images[::-1,::-1,:])
                bps = bp_map[...,cells]
                images[bps != 0] = 0

            # integrate
            t_start = time.time()
            # Momentum is the same for all of the data, hence, we
            # get it once and use it everywhere
            momentum, azi, normalised = integrate(images)
            means = running_mean(normalised, running_averages)
            diffs = differences(normalised, running_averages)
            diffs_integs = diff_integrals(diffs, momentum, running_averages)
            integ_t = time.time() - t_start

            # display
            t_start = time.time()
            self.update_figures(momentum, azi, normalised, 
                                means, diffs, diffs_integs,
                                images, tid)
            plot_t = time.time() - t_start

            print(tid, "retrieval", retrieval_t,
                  "integration", integ_t, "plot", plot_t)

    def stop(self):
        """Stop the update thread"""
        self.running = False
        self.join()


if __name__ == '__main__':
    addr = sys.argv[1] if len(sys.argv) > 1 else "tcp://10.253.0.53:4501"
    client = Client(addr)

    global app
    app = qt.QApplication([])

    updateThread = UpdateThread(client)
    updateThread.setVisible(True)

    updateThread.start()
    app.exec_()
    updateThread.stop()
    sys.exit()
