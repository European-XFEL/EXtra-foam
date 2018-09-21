"""
Offline and online data analysis tool for Azimuthal integration at
FXE instrument, European XFEL.

DAQ module.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time
from threading import Thread

from karabo_bridge import Client

from .data_processing import DataProcessor
from .logging import logger
from .config import DataSource


class DaqWorker(Thread):
    def __init__(self, address, out_queue, source, **kwargs):
        """Initialization."""
        super().__init__()

        self._source = source
        self._address = address
        self._out_queue = out_queue
        self._processor = DataProcessor(**kwargs)
        self._running = True

    def run(self):
        """Override."""
        with Client(self._address) as client:
            while self._running is True:

                t0 = time.perf_counter()

                data = client.next()

                logger.debug("Time for retrieving data from the server: {:.1f} ms"
                             .format(1000 * (time.perf_counter() - t0)))

                t0 = time.perf_counter()

                if self._source == DataSource.CALIBRATED_FILE:
                    processed_data = self._processor.process_calibrated_data(
                        data, from_file=True)
                elif self._source == DataSource.CALIBRATED:
                    processed_data = self._processor.process_calibrated_data(data)
                elif self._source == DataSource.ASSEMBLED:
                    processed_data = self._processor.process_assembled_data(data)
                elif self._source == DataSource.PROCESSED:
                    processed_data = data[0]
                else:
                    raise ValueError("Unknown data source!")

                logger.debug("Time for data processing: {:.1f} ms in total!\n"
                             .format(1000 * (time.perf_counter() - t0)))

                logger.debug("Current queue size: {}".format(self._out_queue.qsize()))

                self._out_queue.put(processed_data)

        with self._out_queue.mutex:
            self._out_queue.queue.clear()

    def terminate(self):
        self._running = False
