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

from karabo_data import ZMQStreamer, RunDirectory, stack_detector_data
from .config import config


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
        self._stream()

    def _stream(self):
        streamer = ZMQStreamer(self._port)
        streamer.start()

        run = RunDirectory(self._folder)
        # TOPIC to be later replaced by Detector type LPD, AGIPD, JFRAU
        if config["TOPIC"] == "FXE" or config["TOPIC"] == "SPB":
            devices = [("*DET/*CH0:xtdf", "image.data")]
        elif config["TOPIC"] == "JungFrau":
            devices = [("*/DET/*:daqOutput", "data.adc")]
        else:
            devices = None

        for tid, data in run.trains(devices=devices, require_all=True):
            if data:
                streamer.feed(data)
        streamer.stop()

    def terminate(self):
        """Override."""
        super().terminate()
