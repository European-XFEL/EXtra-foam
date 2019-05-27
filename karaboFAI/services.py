"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Services.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import argparse
import multiprocessing as mp
import os
import subprocess
import time
import faulthandler
import sys

import redis

from . import __version__
from .config import config
from .logger import logger
from .gui import MainGUI, mkQApp
from .pipeline import Bridge, Scheduler
from .pipeline.worker import ProcessProxy


def check_system_resource():
    """Check the resource of the current system"""
    n_cpus = mp.cpu_count()

    n_gpus = 0

    return n_cpus, n_gpus


def try_to_connect_redis_server(host, port, *, password=None, n_attempts=5):
    """Try to connect to a starting Redis server.

    :param str host: IP address of the redis server.
    :param int port:: Port of the redis server.
    :param str password: Password of the redis server.
    :param int n_attempts: Number of attempts to connect to the redis server.

    Raises:
        ConnectionError: raised if the Redis server cannot be connected.
    """
    # Create a Redis client to check whether the server is reachable.
    client = redis.Redis(host=host, port=port, password=password)

    for i in range(n_attempts):
        try:
            logger.info(f"Say hello to Redis server at {host}:{port}")
            client.ping()
        except redis.ConnectionError:
            time.sleep(1)
            logger.info("No response from the Redis server")
        else:
            logger.info("Received response from the Redis server")
            return

    raise ConnectionError(f"Failed to connect to the Redis server at "
                          f"{host}:{port}.")


def start_redis_server():
    """Start a Redis server.

    :return ProcessInfo: process info.

    Raises:
        FileNotFoundError: raised if the Redis executable does not exist.
    """
    executable = config["REDIS_EXECUTABLE"]
    if not os.path.isfile(executable):
        raise FileNotFoundError

    password = config["REDIS_PASSWORD"]
    host = 'localhost'
    port = config["REDIS_PORT"]

    # Construct the command to start the Redis server.
    command = [executable,
               "--port", str(port),
               "--requirepass", password,
               "--loglevel", "warning"]

    process = subprocess.Popen(command)

    # Create a Redis client just for configuring Redis.
    client = redis.Redis(host, port, password=password)

    try:
        # wait for the Redis server to start
        try_to_connect_redis_server(host, port, password=password)
    except ConnectionError:
        logger.error(f"Unable to start a Redis server at {host}:{port}")
        return False

    time.sleep(0.1)
    if process.poll() is None:
        # Put a time stamp in Redis to indicate when it was started.
        client.set("redis_start_time", time.time())

        logger.info(f"Redis server started at {host}:{port}")
        ProcessProxy().register("redis", process)
    else:
        logger.info(f"Found existing Redis server at {host}:{port}")

    return True


class FAI:
    def __init__(self, detector):
        # update global configuration
        config.load(detector)

        # Redis server must be started at first since when the GUI starts,
        # it needs to write all the configuration into Redis.
        if not start_redis_server():
            sys.exit(0)

        try:
            # process which runs one or more zmq bridge
            self.bridge = Bridge()

            # process which runs the scheduler
            self.scheduler = Scheduler(detector)
            self.scheduler.connect_input(self.bridge)

            self.app = mkQApp()
            self.gui = MainGUI()
        except Exception as e:
            logger.error(repr(e))
            ProcessProxy().terminte_popens()
            sys.exit(0)

    def init(self):
        n_cpus, n_gpus = check_system_resource()
        logger.info(f"Number of available CPUs: {n_cpus}, "
                    f"number of available GPUs: {n_gpus}")

        self.bridge.start()
        self.scheduler.start()

        self.gui.connectInput(self.scheduler)
        self.gui.start_sgn.connect(self.bridge.activate)
        self.gui.stop_sgn.connect(self.bridge.pause)


def application():
    parser = argparse.ArgumentParser(prog="karaboFAI")
    parser.add_argument('-V', '--version', action='version',
                        version="%(prog)s " + __version__)
    parser.add_argument("detector", help="detector name (case insensitive)",
                        choices=[det.upper() for det in config.detectors],
                        type=lambda s: s.upper())
    parser.add_argument('--debug', action='store_true',
                        help="Run in debug mode")

    args = parser.parse_args()

    if args.debug:
        logger.debug("'faulthandler enabled")
    else:
        logger.setLevel("INFO")

    detector = args.detector
    if detector == 'JUNGFRAU':
        detector = 'JungFrau'
    elif detector == 'FASTCCD':
        detector = 'FastCCD'
    elif detector == 'BASLERCAMERA':
        detector = 'BaslerCamera'
    else:
        detector = detector.upper()

    if not faulthandler.is_enabled():
        faulthandler.enable(all_threads=False)

    fai = FAI(detector)

    fai.init()

    mkQApp().exec_()


if __name__ == "__main__":

    application()
