import logging
import signal
import sys
import time

from h5py import File
import numpy as np

from karabo_data import RunDirectory, stack_detector_data, ZMQStreamer
from karabo_data.geometry import LPDGeometry

from constants import bridge_key


logging.basicConfig(level=logging.INFO,
                    format="PID %(process)5s: %(message)s",
                    stream=sys.stderr)
log = logging.getLogger()


class DataServer:
    def __init__(self, port, datapath, geopath):
        self.port = port
        self.path = datapath
        self.server = ZMQStreamer(self.port)

        # The following is valid-ish for the 20180318 geometry
        pos = [(-11.4, -229), (11.5, -8), (-254.5, 16), (-278.5, -275)]
        with File(geopath, 'r') as f:
            self.geometry = LPDGeometry.from_h5_file_and_quad_positions(f, pos)

    def get_detector_data(self) -> (int, np.ndarray):
        """Get the data from the files in the directory, assembled.
        """
        run = RunDirectory(self.path)
        run.info()
        for tid, data in run.trains():
            loadtime = time.time()
            image_data = stack_detector_data(data, "image.data", only="LPD")
            cell_data = stack_detector_data(data, "image.cellId", only="LPD")
            # We squeeze the second dimension which is of size one. FOR NOW??
            image_data = image_data.squeeze()
            image_data, centre = self.geometry.position_all_modules(image_data)

            # Fake being the online preview with funny axes
            image_data = np.moveaxis(image_data, 0, 2)

            # Replace the LPDGeometry-induced NaNs with zeroes
            image_data = np.nan_to_num(image_data)

            data[bridge_key] = {"image.data": image_data,
                                "image.cellId": cell_data,
                                "header.trainId": tid}

            t_ = time.time() - loadtime
            log.info("Loaded {} in {}".format(tid, t_))

            yield tid, data

    def serve(self):
        log.info("Starting serving files")
        getter = self.get_detector_data()
        self.server.start()

        while True:
            try:
                tid, data = next(getter)
                self.server.feed(data)
            except StopIteration:
                break

        log.info("Served all trains")

    def quit(self, sig=None, frame=None):
        log.info(f"Quitting because {sig}")
        self.server.stop()
        sys.exit(0)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        exit("Usage:\n\tpython %s PORT DATAPATH [GEOPATH]" % sys.argv[0])

    _, port, path = sys.argv[:3]

    geopath = "~/data/lpd_mar_18.h5"
    if len(sys.argv) > 3:
        geopath = sys.argv[3]

    server = DataServer(port, path, geopath)

    signal.signal(signal.SIGINT, server.quit)
    signal.signal(signal.SIGTERM, server.quit)

    server.serve()
    server.quit(signal="Served all trains")
