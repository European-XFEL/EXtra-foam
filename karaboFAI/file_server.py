"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

File server.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from multiprocessing import Process

from karabo_data import serve_files


class FileServer(Process):
    """Stream the file data in another process."""
    def __init__(self, folder, port):
        """Initialization."""
        super().__init__()
        self._folder = folder
        self._port = port

        self._running = True

    def run(self):
        """Override."""
        serve_files(self._folder, self._port)

    def terminate(self):
        """Override."""
        super().terminate()
