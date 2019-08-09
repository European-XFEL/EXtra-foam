"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

File server.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import datetime
import os.path as osp
from multiprocessing import Process
import re
from time import time

from karabo_data import RunDirectory, ZMQStreamer

from ..config import config
from ..utils import profiler
from ..logger import logger


def get_info(run):
    """karabo data info method"""
    DETECTOR_SOURCE_RE = re.compile(r'(.+)/DET/(\d+)CH')
    info = ""
    first_train = run.train_ids[0]
    last_train = run.train_ids[-1]
    train_count = len(run.train_ids)
    span_sec = (last_train - first_train) / 10
    span_txt = str(datetime.timedelta(seconds=span_sec))

    detector_modules = {}
    for source in run.detector_sources:
        name, modno = DETECTOR_SOURCE_RE.match(source).groups((1, 2))
        detector_modules[(name, modno)] = source

    # A run should only have one detector, but if that changes, don't hide it
    detector_name = ','.join(sorted(set(k[0] for k in detector_modules)))

    info += f'\n# of trains:   {train_count}'
    info += f'\nDuration:      {span_txt}'
    info += f'\nFirst train ID: {first_train}'
    info += f'\nLast train ID: {last_train}\n'

    info += f'\n{len(run.detector_sources)} detector modules ({detector_name})'
    if len(detector_modules) > 0:
        # Show detail on the first module (the others should be similar)
        mod_key = sorted(detector_modules)[0]
        mod_source = detector_modules[mod_key]
        dinfo = run.detector_info(mod_source)
        module = ' '.join(mod_key)
        dims = ' x '.join(str(d) for d in dinfo['dims'])
        info += f"\n  e.g. module {module} : {dims} pixels"
        info += (f"\n  {dinfo['frames_per_train']} frames per train, "
                 f" {dinfo['total_frames']} total frames")

    info += "\n"
    non_detector_inst_srcs = run.instrument_sources - run.detector_sources
    info += (f"\n{len(non_detector_inst_srcs)} instrument sources "
             f"(excluding detectors):")

    for d in sorted(non_detector_inst_srcs):
        info += f"\n  -{d}"
    info += "\n"
    info += f"\n{len(run.control_sources)} control sources:"
    for d in sorted(run.control_sources):
        info += f"\n  -{d}"
    return info


@profiler("Gather sources")
def gather_sources(path):
    """Gather slow sources from a run

    Parameters:
    -----------
    path: str
          Path to HDF5 run folder

    Return:
    -------
    Slow sources: Frozenset
        Set of slow sources available. Empty if none found.
    Run info: string
        Information about the run. Empty string if no info found
    """
    slow = frozenset()
    info = ""
    if osp.isdir(path):
        try:
            # This will work in case users are using data stored
            # in /gpfs/exfel/exp/INSTRUMENT/cycle/proposal/proc/runumber
            # or they have raw folder with same path instead of 'proc'
            # in it in the end.
            run = RunDirectory(path.replace('/proc/', '/raw/'))
            slow = run.control_sources
            info = get_info(run)
        except Exception as ex:
            # Will be raised if no folder with 'raw' exist or no files
            # found in raw folder.
            logger.info("Failed to find 'raw' folder! "
                        "Looking for slow data in current folder")
            # Fallback to corrected datapath. Will return empty
            # frozenset if no control source found.
            try:
                run = RunDirectory(path)
                slow = run.control_sources
                info = get_info(run)
            except Exception as ex:
                logger.error(repr(ex))

    return slow, info


def generate_meta(sources, tid):
    """Generate metadata in case of repeat stream"""

    time_now = int(time() * 10**18)
    sec = time_now // 10**18
    frac = time_now % 10**18
    meta = {key:
            {'source': key, 'timestamp.sec': sec, 'timestamp.frac': frac,
             'timestamp.tid': tid} for key in sources}
    return meta


def serve_files(path, port, slow_devices=None, fast_devices=None,
                require_all=False, repeat_stream=False, **kwargs):
    """Stream data from files through a TCP socket.

    Parameters
    ----------
    path: str
        Path to the HDF5 file or file folder.
    port: int
        Local TCP port to bind socket to.
    slow_devices: list of tuples
        [('src', 'prop')]
    fast_devices: list of tuples
        [('src', 'prop')]
    require_all: bool
        If set to True, will stream only trainIDs that has data
        corresponding to keys specified in fast_devices.
        Default: False
    repeat_stream: bool
        If set to True, will continue streaming when trains()
        iterator is empty. Trainids will be monotonically increasing.
        Default: False
    """
    raw_data = None
    try:
        corr_data = RunDirectory(path)
        num_trains = len(corr_data.train_ids)
    except Exception as ex:
        logger.error(repr(ex))
        return

    if slow_devices:
        # slow_devices is not None only when some slow data source was
        # selected. That means raw_data atleast was same as corr_data
        try:
            raw_data = RunDirectory(path.replace('/proc/', '/raw/'))
        except Exception as ex:
            raw_data = corr_data

    streamer = ZMQStreamer(port, **kwargs)
    streamer.start()

    counter = 0

    while True:
        for tid, train_data in corr_data.trains(devices=fast_devices,
                                                require_all=require_all):
            # loop over corrected DataCollection
            if raw_data is not None:
                try:
                    # get raw data corresponding to the train id : tid
                    _, raw_train_data = raw_data.train_from_id(
                        tid, devices=slow_devices)
                    # Merge corrected and raw data for train id: tid
                    train_data = {**raw_train_data, **train_data}
                except ValueError:
                    # Value Error is raised by karabo data when raw data
                    # corresponding to slow devices is not found.
                    pass
            if train_data:
                # Generate fake meta data with monotically increasing
                # trainids only after the actual trains in corrected data
                # are exhausted
                meta = generate_meta(
                    train_data.keys(), tid+counter) if counter > 0 else None
                streamer.feed(train_data, metadata=meta)
        if not repeat_stream:
            break
        # increase the counter by total number of trains in a run
        counter += num_trains

    streamer.stop()


class FileServer(Process):
    """Stream the file data in another process."""

    def __init__(self, folder, port, detector=None,
                 slow_devices=None, repeat_stream=False):
        """Initialization."""
        super().__init__()
        self._detector = detector
        self._folder = folder
        self._port = port
        self._slow_devices = slow_devices
        self._repeat_stream = repeat_stream

    def run(self):
        """Override."""
        detector = \
            self._detector if self._detector is not None else config["DETECTOR"]

        if detector in ["LPD", "AGIPD", "DSSC"]:
            fast_devices = [("*DET/*CH0:xtdf", "image.data")]
        elif detector == "JungFrau":
            fast_devices = [("*/DET/*:daqOutput", "data.adc")]
        elif detector == "FastCCD":
            fast_devices = [("*/DAQ/*:daqOutput", "data.image.pixels")]
        else:
            raise NotImplementedError(f"Unknown Detector: {detector}")

        serve_files(self._folder, self._port, slow_devices=self._slow_devices,
                    fast_devices=fast_devices, require_all=True,
                    repeat_stream=self._repeat_stream)
