"""
Offline and online data analysis tool for Azimuthal integration at
FXE instrument, European XFEL.

File server module.

Author: Jun Zhu, jun.zhu@xfel.eu, zhujun981661@gmail.com
Copyright (C) European XFEL GmbH Hamburg. All rights reserved.
"""
from multiprocessing import Process

from karabo_data import serve_files

from .logging import logger


class FileServer(Process):
    """Stream the file data in another process."""
    def __init__(self, folder, port):
        super().__init__()
        self._folder = folder
        self._port = port

        self._running = True

    def run(self):
        """Serve the file."""
        serve_files(self._folder, self._port)

    def terminate(self):
        super().terminate()
        logger.info("File serving was terminated!")
