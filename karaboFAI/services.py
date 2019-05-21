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
import sys
import subprocess
import time
import faulthandler

import redis

from zmq.error import ZMQError

from . import __version__
from .config import config, redis_connection
from .metadata import Metadata as mt
from .metadata import MetaProxy
from .logger import logger
from .gui import MainGUI, mkQApp
from .offline import FileServer
from .pipeline import Bridge, ProcessInfo, Scheduler


def check_system_resource():
    """Check the resource of the current system"""
    n_cpus = mp.cpu_count()

    n_gpus = 0

    return n_cpus, n_gpus


def try_to_connect_redis_server(host, port, *, password=None, n_attempts=10):
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

    # try 10 times
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

    :returns: port, process info
    :rtype: int, ProcessInfo

    Raises:
        FileNotFoundError: raised if the Redis executable does not exist.
    """
    redis_cfg = config["REDIS"]
    executable = redis_cfg["EXECUTABLE"]
    if not os.path.isfile(executable):
        raise FileNotFoundError

    port = redis_cfg["PORT"]
    password = redis_cfg["PASSWORD"]

    # Construct the command to start the Redis server.
    command = [executable]
    command += (["--port", str(port),
                 "--requirepass", password,
                 "--loglevel", "warning"])

    process = subprocess.Popen(command)

    # Create a Redis client just for configuring Redis.
    client = redis.Redis("localhost", port, password=password)

    # wait for the Redis server to start
    try_to_connect_redis_server("localhost", port, password=password)

    # Put a time stamp in Redis to indicate when it was started.
    client.set("redis_start_time", time.time())

    logger.info(f"\nRedis servert started at '127.0.0.1':{port}")

    return ProcessInfo(
        process=process,
        stdout_file=None,
        stderr_file=None,
    )


class Fai:
    """Fai class.

    It manages all services in karaboFAI: QApplication, Redis, Processors,
    etc.
    """

    def __init__(self, detector, *, redis_port=-1):
        """Initialization."""
        n_cpus, n_gpus = check_system_resource()
        logger.info(f"Number of available CPUs: {n_cpus}, "
                    f"number of available GPUs: {n_gpus}")

        # update global configuration
        config.load(detector, redis_port=redis_port)

        mkQApp()
        self._gui = MainGUI()

        self._redis_process_info = None

        # process which runs one or more zmq bridge
        self._bridge = Bridge()

        # process which runs the scheduler
        self._scheduler = Scheduler()
        self._scheduler.connect_input(self._bridge)

        # a file server which streams data from files
        self._file_server = None

    def _shutdown_redis_server(self):
        logger.info("Shutting down the Redis server...")
        self._redis_process_info.process.terminate()

    def _shutdown_scheduler(self):
        logger.info("Shutting down the scheduler...")
        self._scheduler.shutdown()
        self._scheduler.join()
        logger.info("Scheduler has been shutdown!")

    def _shutdown_bridge(self):
        logger.info("Shutting down the bridge...")
        self._bridge.shutdown()
        self._bridge.join()
        logger.info("Bridge has been shutdown!")

    def shutdown(self):
        self._stop_fileserver()

        self._shutdown_bridge()
        self._shutdown_scheduler()

        self._shutdown_redis_server()

    def _start_fileserver(self):
        cfg = redis_connection().hgetall(mt.DATA_SOURCE)
        folder = cfg['data_folder']
        port = cfg['endpoint'].split(':')[-1]
        # process can only be start once
        self._file_server = FileServer(folder, port)
        try:
            # TODO: signal the end of file serving
            self._file_server.start()
            logger.info("Start serving file in the folder {} through port {}"
                        .format(folder, port))
        except FileNotFoundError:
            logger.info("{} does not exist!".format(folder))
            return
        except ZMQError:
            logger.info("Port {} is already in use!".format(port))
            return

        Mediator().file_server_started_sgn.emit()

    def _stop_fileserver(self):
        if self._file_server is not None and self._file_server.is_alive():
            # fileserver does not have any shared object
            self._file_server.terminate()

        if self._file_server is not None:
            self._file_server.join()

        Mediator().file_server_stopped_sgn.emit()

    def init(self):
        if not faulthandler.is_enabled():
            faulthandler.enable(all_threads=False)

        self._redis_process_info = start_redis_server()

        MetaProxy().reset()

        self._gui._mediator.start_file_server_sgn.connect(self._start_fileserver)
        self._gui._mediator.stop_file_server_sgn.connect(self._stop_fileserver)

        self._gui.connectInput(self._scheduler)

        self._gui.start_sgn.connect(self._bridge.activate)
        self._gui.stop_sgn.connect(self._bridge.pause)

        self._bridge.start()
        self._scheduler.start()


def application():
    parser = argparse.ArgumentParser(prog="karaboFAI")
    parser.add_argument('-V', '--version', action='version',
                        version="%(prog)s " + __version__)
    parser.add_argument("detector", help="detector name (case insensitive)",
                        choices=[det.upper() for det in config.detectors],
                        type=lambda s: s.upper())
    parser.add_argument('--debug', action='store_true',
                        help="Run in debug mode")
    parser.add_argument('--redis_port', type=int, default=-1,
                        help="Port for start the redis server.")

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

    fai = Fai(detector, redis_port=args.redis_port)

    try:
        fai.init()

        mkQApp().exec_()
    finally:
        fai.shutdown()


if __name__ == "__main__":

    application()
