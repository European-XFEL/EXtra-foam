import time
from threading import Thread

from .data_processing import DataProcessor
from .logger import log
from .config import DataSource


class DaqWorker(Thread):
    def __init__(self, client, out_queue, source, *, geom_file=None):
        super().__init__()

        self._source = source
        self._client = client
        self._out_queue = out_queue
        self._processor = DataProcessor(geom_file)
        self._running = True

    def run(self):
        while self._running is True:
            # retrieve
            t0 = time.perf_counter()

            data = self._client.next()

            log.debug("Time for retrieving data from the server: {:.1f} ms"
                      .format(1000 * (time.perf_counter() - t0)))

            t0 = time.perf_counter()

            if self._source == DataSource.CALIBRATED:
                processed_data = self._processor.process_calibrated_data(data)
            elif self._source == DataSource.ASSEMBLED:
                processed_data = self._processor.process_assembled_data(data)
            elif self._source == DataSource.PROCESSED:
                processed_data = data
            else:
                raise ValueError("Unknown data source!")

            # this information will flood the GUI logger window and
            # crash the GUI.
            log.debug("Total time for processing the data: {:.1f} ms"
                      .format(1000 * (time.perf_counter() - t0)))

            self._out_queue.append(processed_data)

        self._out_queue.clear()

    def terminate(self):
        self._running = False
