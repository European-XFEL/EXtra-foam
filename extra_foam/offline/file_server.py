"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp
import re
import random
from time import time, sleep
from collections import deque

from extra_data import by_id, RunDirectory
from extra_data.export import ZMQStreamer

from .offline_config import StreamMode
from ..utils import profiler


_ALL_PROPERTIES = "*"


def run_info(rd):
    """Return the basic information of a run.

    :param DataCollection rd: the run data.
    """
    if rd is None:
        return 0, -1, -1

    first_train = rd.train_ids[0]
    last_train = rd.train_ids[-1]
    n_trains = len(rd.train_ids)

    return n_trains, first_train, last_train


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
        "actualEnergy.value",
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


def serve_files(run_data, port, shared_tid, shared_rate, max_rate,
                tid_range=None,
                mode=StreamMode.NORMAL,
                detector_sources=None,
                instrument_sources=None,
                control_sources=None,
                require_all=True,
                repeat_stream=False,
                buffer_size=2,
                **kwargs):
    """Stream data from files through a TCP socket.

    Parameters
    ----------
    run_data: list/tuple
        A list/tuple of calibrated and raw run data.
    port: int
        Local TCP port to bind socket to.
    shared_tid: Value
        The latest streamed train ID shared between processes.
    shared_rate: Value
        The actual streaming rate in Hz.
    max_rate: float
        The maximum streaming rate in Hz.
    tid_range: tuple
        (start, end, step) train ID.
    mode: StreamMode
        The stream mode:
        - Normal: sources in a train are streamed together;
        - Random shuffle: sources in a train are streamed one by one
                          and the order is random.

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
    buffer_size: int
        ZMQStreamer buffer size.
    """
    rd_cal, rd_raw = run_data
    num_trains = len(rd_cal.train_ids)

    if rd_raw is None:
        rd_raw = rd_cal

    streamer = ZMQStreamer(port, maxlen=buffer_size, **kwargs)
    streamer.start()  # run "REP" socket in a thread

    train_delay = 1 / max_rate
    counter = 0
    n_buffered = 0
    t_sent = deque(maxlen=10)
    while True:
        for tid, train_data in rd_cal.trains(
                devices=detector_sources,
                train_range=by_id[tid_range[0]:tid_range[1]:tid_range[2]],
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
                if mode == StreamMode.NORMAL:
                    # Generate fake meta data with monotonically increasing
                    # train IDs only after the actual trains in corrected data
                    # are exhausted
                    meta = generate_meta(
                        train_data.keys(), tid+counter) if counter > 0 else None
                    streamer.feed(train_data, metadata=meta)
                elif mode == StreamMode.RANDOM_SHUFFLE:
                    keys = list(train_data.keys())
                    random.shuffle(keys)
                    for k in keys:
                        meta = generate_meta(
                            [k], tid+counter) if counter > 0 else None
                        streamer.feed({k: train_data[k]}, metadata=meta)

            if n_buffered <= buffer_size:
                # Do not count the first trains which are buffered immediately but
                # may not be sent.
                n_buffered += 1
            else:
                t_sent.append(time())

            # update processing rate
            n = len(t_sent) - 1
            if n > 0:
                inter_train_delay = t_sent[-1] - t_sent[-2]
                sleep_delay = train_delay - inter_train_delay

                # If the delay is greater than some epsilon, sleep
                if sleep_delay > 0.001:
                    sleep(sleep_delay)

                    # Record the current time as the end of processing time for
                    # this train.
                    t_sent.pop()
                    t_sent.append(time())

                shared_rate.value = n / (t_sent[-1] - t_sent[0])

            # update the train ID just sent
            shared_tid.value = tid

        if not repeat_stream:
            break

        # increase the counter by total number of trains in a run
        counter += num_trains

    streamer.stop()
