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
import os
import time
import faulthandler
import sys

import psutil

import redis

from . import __version__
from .config import config
from .logger import logger
from .gui import MainGUI, mkQApp
from .pipeline import Scheduler
from .pipeline.worker import ProcessInfo, register_fai_process
from .utils import check_system_resource
from .ipc import redis_connection


_N_CPUS, _N_GPUS, _SYS_MEMORY = check_system_resource()


def try_to_connect_redis_server(host, port, *, n_attempts=5):
    """Try to connect to a starting Redis server.

    :param str host: IP address of the redis server.
    :param int port:: Port of the redis server.
    :param int n_attempts: Number of attempts to connect to the redis server.

    Raises:
        ConnectionError: raised if the Redis server cannot be connected.
    """
    client = redis_connection()

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
        ConnectionError: raised if failed to start Redis server.
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

    process = psutil.Popen(command)

    # The global redis client is created
    client = redis_connection()

    try:
        # wait for the Redis server to start
        try_to_connect_redis_server(host, port)
    except ConnectionError:
        logger.error(f"Unable to start a Redis server at {host}:{port}")
        raise

    time.sleep(0.1)
    if process.poll() is None:
        # Put a time stamp in Redis to indicate when it was started.
        client.set(f"{config['DETECTOR']}:redis_start_time", time.time())

        logger.info(f"Redis server started at {host}:{port}")

        register_fai_process(ProcessInfo(name="redis",
                                         process=process))

        try:
            mem_frac = config["REDIS_MAX_MEMORY_FRAC"]
            if mem_frac < 0.001 or mem_frac > 0.5:
                mem_frac = 0.3
            client.config_set("maxmemory", int(mem_frac * _SYS_MEMORY))
            mem_in_bytes = int(client.config_get('maxmemory')['maxmemory'])
            logger.info(f"Redis memory is capped at "
                        f"{mem_in_bytes / 1024 ** 3:.1f} GB")
        except Exception as e:
            logger.error(f"Failed to config the Redis server.\n" + repr(e))
            sys.exit(0)

    else:
        logger.info(f"Found existing Redis server at {host}:{port}")


def health_check():
    residual = []
    for proc in psutil.process_iter(attrs=["name", "pid", "cmdline"]):
        if proc.info['pid'] == os.getpid():
            continue

        if 'redis-server' in proc.info['name'] or \
                'karaboFAI.services' in proc.info['cmdline']:
            residual.append(proc)

    if residual:
        ret = input(
            "Warning: Found old karaboFAI instance(s) running in this "
            "machine!!!\n\n"
            "Running more than two karaboFAI instances with the same \n"
            "detector can result in undefined behavior. You can try to \n"
            "kill the other instances if it is owned by you. \n"
            "Note: you are not able to kill other users' instances! \n\n"
            "Send SIGKILL? (y/n)")

        if ret.lower() == 'y':
            for p in residual:
                p.kill()

        gone, alive = psutil.wait_procs(residual, timeout=1.0)
        if alive:
            for p in alive:
                print(f"process {p} survived SIGKILL, "
                      f"please contact the user: {p.username()}")
        else:
            print("Residual processes have been terminated!!!")


class FAI:
    def __init__(self, detector):

        # update global configuration
        config.load(detector)

        # Redis server must be started at first since when the GUI starts,
        # it needs to write all the configuration into Redis.
        try:
            start_redis_server()
        except (ConnectionError, FileNotFoundError):
            sys.exit(1)

        try:
            # process which runs the scheduler
            self.scheduler = Scheduler(detector)

            self.app = mkQApp()
            self.gui = MainGUI(start_thread_logger=True)
        except Exception as e:
            logger.error(repr(e))
            sys.exit(1)

    def init(self):
        logger.info(f"Number of available CPUs: {_N_CPUS}, "
                    f"number of available GPUs: {_N_GPUS}, "
                    f"total system memory: {_SYS_MEMORY/1024**3:.1f} GB")

        self.scheduler.start()
        register_fai_process(ProcessInfo(name='scheduler',
                                         process=self.scheduler))

        self.gui.connectInputToOutput(self.scheduler.output)
        self.gui.start_sgn.connect(self.start)
        self.gui.stop_sgn.connect(self.pause)

    def start(self):
        self.scheduler.resume()

    def pause(self):
        self.scheduler.pause()


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

    health_check()

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
