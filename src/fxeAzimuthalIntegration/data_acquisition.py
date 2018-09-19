"""
Offline and online data analysis tool for Azimuthal integration at
FXE instrument, European XFEL.

DAQ module.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time

from .logging import logger


class DaqWorker:
    def __init__(self, client):
        """Initialization."""
        super().__init__()

        self._client = client

        self._running = False

    def run(self, out_queue):
        self._running = True
        while self._running:
            t0 = time.perf_counter()

            out_queue.put(self._client.next())

            logger.debug("Time for retrieving data from the server: {:.1f} ms"
                         .format(1000 * (time.perf_counter() - t0)))

    def terminate(self):
        self._running = False
