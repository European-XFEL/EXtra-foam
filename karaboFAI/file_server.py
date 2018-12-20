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

from .config import config

# Temporary re-implementation of serve_files from karabo_data.
from karabo_data import H5File, RunDirectory, ZMQStreamer
import os.path as osp

def serve_files(path, port, devices=None, require_all=False, **kwargs):
    """Stream data from files through a TCP socket.

    Parameters
    ----------
    path: str
        Path to the HDF5 file or file folder.
    port: int
        Local TCP port to bind socket to.
    devices: list, dict
        3 possible ways to select data. See :meth:`select`
    require_all: bool
        If set to True, will stream only trainIDs that has data
        corresponding to keys specified in devices.
        Default: False
    """
    if osp.isdir(path):
        data = RunDirectory(path)
    else:
        data = H5File(path)

    streamer = ZMQStreamer(port, **kwargs)
    streamer.start()
    for tid, train_data in data.trains(devices=devices,
                                       require_all=require_all):
        if train_data:
            streamer.feed(train_data)
    streamer.stop()

# Once MR is accepted in karabo_data
# from karabo_data import serve_files

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
        if config["TOPIC"] == "FXE" or config["TOPIC"] == "SPB":
            devices = [("*DET/*CH0:xtdf", "image.data")]
        elif config["TOPIC"] == "JungFrau":
            devices = [("*/DET/*:daqOutput", "data.adc")]
        else:
            devices = None

        serve_files(self._folder, self._port, devices=devices, require_all=True)

    def terminate(self):
        """Override."""
        super().terminate()
