"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

File server manager.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from zmq.error import ZMQError

from .file_server import FileServer
from ..gui.mediator import Mediator
from ..metadata import Metadata as mt
from ..metadata import MetaProxy
from ..logger import logger


class FileServerManager:
    def __init__(self):
        self._file_server = None

        self._mediator = Mediator()
        self._meta = MetaProxy()

        self._mediator.start_file_server_sgn.connect(self._start_fileserver)
        self._mediator.stop_file_server_sgn.connect(self._stop_fileserver)

    def _start_fileserver(self):
        cfg = self._meta.get_all(mt.DATA_SOURCE)
        try:
            folder = cfg['data_folder']
            port = cfg['endpoint'].split(':')[-1]
        except KeyError as e:
            logger.error(repr(e))
            return

        # process can only be start once
        self._file_server = FileServer(folder, port)
        try:
            self._file_server.start()
            logger.info("Serving file in the folder {} through port {}"
                        .format(folder, port))
        except FileNotFoundError:
            logger.info("{} does not exist!".format(folder))
            return
        except ZMQError:
            logger.info("Port {} is already in use!".format(port))
            return

        self._mediator.file_server_started_sgn.emit()

    def _stop_fileserver(self):
        if self._file_server is not None and self._file_server.is_alive():
            # a file_server does not have any shared object
            self._file_server.terminate()

        if self._file_server is not None:
            # this join may be redundant
            self._file_server.join()

        self._mediator.file_server_stopped_sgn.emit()

    def shutdown(self):
        self._stop_fileserver()
