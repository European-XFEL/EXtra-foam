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
from queue import Full

from karabo_bridge import Client

from .logging import logger


class DaqWorker(Thread):
    def __init__(self, address, out_queue):
        """Initialization."""
        super().__init__()

        self._address = address
        self._out_queue = out_queue
        self._running = True

    def run(self):
        """Override."""
        logger.debug("Start data acquisition...")
        with Client(self._address) as client:
            while self._running is True:

                t0 = time.perf_counter()

                data = client.next()

                logger.debug("Time for retrieving data from the server: {:.1f} ms"
                             .format(1000 * (time.perf_counter() - t0)))

                try:
                    self._out_queue.put(data, timeout=5)
                except Full:
                    continue

    def terminate(self):
        self._running = False
