from math import sqrt
import sys
import threading

from karabo_bridge import Client
from silx.gui import qt

from data_processing import process_data
from plots import get_figures
import config as cfg


class UpdateThread(threading.Thread, qt.QMainWindow):
    def __init__(self, bridge_client):
        """
        :param client: A KaraboBridge client.
        """
        super(UpdateThread, self).__init__()
        self.__init_window()

        self.client = bridge_client
        self.running = False
        self.first_loop = True

    def __init_window(self):
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

    def update_figures(self, kb_data):
        """

        :param tuple kb_data: (data, metadata)
        """
        data = process_data(kb_data)
        if data is None:
            return

        # plot results
        title = ("Azimuthal Integration over {} pulses {}"
                 "".format(len(data["intensity"]), data["tid"]))
        self.setWindowTitle(title)

        # assert len(data["azi"]) == len(data["normalised"])
        #
        for i, intensity in enumerate(data["intensity"]):
            # norm_scattering = data["normalised"][i]
            self.integ.addCurveThreadSafe(data["momentum"],
                                          intensity,
                                          legend=str(i),
                                          copy=False,
                                          resetzoom=self.first_loop)
        #
        #     self.normalised.addCurveThreadSafe(data["momentum"],
        #                                        norm_scattering,
        #                                        legend=str(index),
        #                                        copy=False,
        #                                        resetzoom=self.first_loop)

        # Red is the difference between the running average and the
        # current pulse, pink is the running mean, and blue the current
        # pulse
        for idx in cfg.RUNNING_AVERAGES:
            p = getattr(self, "p{}".format(idx))
            p.addCurveThreadSafe(data["momentum"],
                                 data["intensity"].mean(axis=0),
                                 legend="mean",
                                 copy=False,
                                 color="#FF0000")
            # p.addCurveThreadSafe(data["momentum"],
            #                      data["diffs"][idx],
            #                      legend="diff",
            #                      copy=False,
            #                      color="#FF9099")
        #     p.addCurveThreadSafe(data["momentum"], data["normalised"][idx],
        #                          legend="pulse", copy=False,
        #                          resetzoom=False, color="#0000FF")
        #
        #     p = getattr(self, "pinteg{}".format(idx))
        #     p.addPointThreadSafe(data["diffs_integs"][idx])
        #
        # # plot the image from the detector
        # pulse_id = random.randint(0, len(data["azi"])-1)
        # title = "Current train: {}".format(data["tid"])
        # title += "\nPulse previewed: {}".format(pulse_id)
        # self.plt2d.setGraphTitle(title)
        # image = data["images"][..., pulse_id]
        # self.plt2d.addImage(image, replace=True, copy=False, yInverted=True)

    def run(self):
        """Method implementing thread loop that gets data,
           integrates, and plots
        """
        bp_map = None
        if cfg.BP_MAP_FILE:
            import h5py
            f = h5py.File(cfg.BP_MAP_FILE, "r")
            bp_map = f["/MappedBadPixels"][()][::-1,::-1,:]
            f.close()
            print("Using bad pixel mask of shape {}".format(bp_map.shape))

        self.running = True
        while self.running:
            # retrieve
            kb_data = self.client.next()
            self.update_figures(kb_data)

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
