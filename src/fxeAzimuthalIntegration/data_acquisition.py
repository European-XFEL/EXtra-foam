"""
Offline and online data analysis tool for Azimuthal integration at
FXE instrument, European XFEL.

DAQ module.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""


class DaqWorker:
    def __init__(self, client):
        """Initialization."""
        super().__init__()

        self._client = client

        self._running = False

    def run(self, out_queue):
        self._running = True
        while self._running:
            data = self._client.next()
            out_queue.put(data)

    def terminate(self):
        self._running = False
