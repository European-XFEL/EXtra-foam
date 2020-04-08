"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp
import re
from time import time

from extra_data import RunDirectory
from extra_data.export import ZMQStreamer

from ..utils import profiler


_ALL_PROPERTIES = "*"


def run_info(rd):
    """Return the basic information of a run.

    :param DataCollection rd: the run data.
    """
    if rd is None:
        return ""

    first_train = rd.train_ids[0]
    last_train = rd.train_ids[-1]
    train_count = len(rd.train_ids)

    info = f'First train ID: {first_train} ' \
           f'/ Last train ID: {last_train} ' \
           f'/ Train ID span: {train_count}'

    return info


@profiler("Load run directories")
def load_runs(path):
    """Load data from both calibrated and raw run directories.

    :param str path: path of the calibrated run directory.

    :raises (ValueError, Exception)
    """
    if not path:
        raise ValueError(f"Empty path!")
    if not osp.isdir(path):
        raise ValueError(f"{path} is not an existing directory!")

    rd_cal = RunDirectory(path)

    try:
        rd_raw = RunDirectory(path.replace('/proc/', '/raw/'))
    except Exception:
        rd_raw = None

    return rd_cal, rd_raw


def _sorted_properties(orig, prioritized):
    """Return the sorted properties.

    :param set orig: original properties.
    :param list prioritized: properties to be prioritized.

    The properties will be sorted by:
    1. sorted prioritized properties;
    2. "*";
    3. the rest properties.
    """
    ret = []
    for ppt in prioritized:
        if ppt in orig:
            ret.append(ppt)
    ret.append('*')
    for ppt in sorted(orig):
        if ppt not in ret:
            ret.append(ppt)

    return ret


def gather_sources(rd_cal, rd_raw):
    """Gather device IDs and properties from run data.

    :param DataCollection rd_cal: Calibrated run data.
    :param DataCollection rd_raw: Raw run data.

    :return: A tuple of dictionaries with keys being the device IDs /
        output channels and values being a list of available properties.
        (detector sources, instrument sources, control sources)
    """
    _prioritized_detector_ppts = sorted([
        "image.data",  # "AGIPD", "LPD", "DSSC"
        "data.adc",  # JungFrau
        "data.image.pixels",  # FastCCD, ePix100
    ])

    _prioritized_instrument_ppts = sorted([
        "data.intensityTD",
        "data.intensitySa1TD",
        "data.intensitySa2TD",
        "data.intensitySa3TD",
        "data.image.pixels",  # Basler camera
    ])

    _prioritized_control_ppts = sorted([
        "actualPosition.value",
        "actualCurrent.value",
        "pulseEnergy.photonFlux.value",
        "beamPosition.ixPos.value",
        "beamPosition.iyPos.value",
    ])

    detector_srcs, instrument_srcs, control_srcs = dict(), dict(), dict()
    if rd_cal is not None:
        for src in rd_cal.instrument_sources:
            if re.compile(r'(.+)/DET/(.+):(.+)').match(src):
                detector_srcs[src] = _sorted_properties(
                    rd_cal.keys_for_source(src), _prioritized_detector_ppts)

        rd = rd_cal if rd_raw is None else rd_raw

        for src in rd.instrument_sources:
            if src not in detector_srcs:
                instrument_srcs[src] = _sorted_properties(
                    rd.keys_for_source(src), _prioritized_instrument_ppts)

        for src in rd.control_sources:
            filtered_ppts = set()
            for ppt in rd.keys_for_source(src):
                # ignore *.timestamp properties and remove ".value" suffix
                if ppt[-5:] == "value":
                    filtered_ppts.add(ppt)
            control_srcs[src] = _sorted_properties(
                filtered_ppts, _prioritized_control_ppts)

    return detector_srcs, instrument_srcs, control_srcs


def generate_meta(devices, tid):
    """Generate metadata in case of repeating stream.

    :param iterable devices: a list of device IDs.
    :param int tid: train ID.
    """
    time_now = int(time() * 10**18)
    sec = time_now // 10**18
    frac = time_now % 10**18
    meta = {key:
            {'source': key, 'timestamp.sec': sec, 'timestamp.frac': frac,
             'timestamp.tid': tid} for key in devices}
    return meta


def serve_files(run_data, port, *,
                detector_sources=None,
                instrument_sources=None,
                control_sources=None,
                require_all=True,
                repeat_stream=False, **kwargs):
    """Stream data from files through a TCP socket.

    Parameters
    ----------
    run_data: list/tuple
        A list/tuple of calibrated and raw run data.
    port: int
        Local TCP port to bind socket to.
    detector_sources: list of tuples
        [('device ID/output channel name', 'property')]
    instrument_sources: list of tuples
        [('device ID/output channel name', 'property')]
    control_sources: list of tuples
        [('device ID/output channel name', 'property')]
    require_all: bool
        If set to True, will stream only trainIDs that has data
        corresponding to keys specified in detector_sources.
        Default: True
    repeat_stream: bool
        If set to True, will continue streaming when trains()
        iterator is empty. Trainids will be monotonically increasing.
        Default: False
    """
    rd_cal, rd_raw = run_data
    num_trains = len(rd_cal.train_ids)

    if rd_raw is None:
        rd_raw = rd_cal

    streamer = ZMQStreamer(port, **kwargs)
    streamer.start()

    counter = 0
    while True:
        for tid, train_data in rd_cal.trains(devices=detector_sources,
                                             require_all=require_all):
            if rd_raw is not None:
                try:
                    # get raw data corresponding to the train id
                    _, raw_train_data = rd_raw.train_from_id(
                        tid, devices=instrument_sources + control_sources)
                    # Merge calibrated and raw data
                    train_data.update(raw_train_data)
                except ValueError:
                    # Value Error is raised by EXtra-data when any raw data
                    # is not found.
                    pass

            if train_data:
                # Generate fake meta data with monotonically increasing
                # train IDs only after the actual trains in corrected data
                # are exhausted
                meta = generate_meta(
                    train_data.keys(), tid+counter) if counter > 0 else None
                streamer.feed(train_data, metadata=meta)

        if not repeat_stream:
            break

        # increase the counter by total number of trains in a run
        counter += num_trains

    streamer.stop()
