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
import traceback
import itertools

import psutil

import redis

from . import __version__
from .config import config
from .ipc import redis_connection, reset_redis_connections
from .logger import logger
from .gui import MainGUI, mkQApp
from .pipeline import ImageWorker, Scheduler
from .processes import ProcessInfo, register_fai_process
from .utils import check_system_resource, query_yes_no
from .gui.windows import FileStreamControllerWindow, ImageToolWindow

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


def start_redis_client():
    """Start a Redis client."""
    exec_ = config["REDIS_EXECUTABLE"].replace("redis-server", "redis-cli")

    if not os.path.isfile(exec_):
        raise FileNotFoundError

    command = [exec_]
    command.extend(sys.argv[1:])

    logger.setLevel("INFO")

    proc = psutil.Popen(command)
    proc.wait()


def start_redis_server():
    """Start a Redis server.

    :return ProcessInfo: process info.

    Raises:
        FileNotFoundError: raised if the Redis executable does not exist.
        ConnectionError: raised if failed to start Redis server.
    """
    reset_redis_connections()

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
        if query_yes_no(
            "Warning: Found old karaboFAI instance(s) running in this "
            "machine!!!\n\n"
            "Running more than two karaboFAI instances with the same \n"
            "detector can result in undefined behavior. You can try to \n"
            "kill the other instances if it is owned by you. \n"
            "Note: you are not able to kill other users' instances! \n\n"
            "Send SIGKILL?"
        ):
            try:
                for p in residual:
                    p.kill()
            except psutil.AccessDenied:
                pass

        gone, alive = psutil.wait_procs(residual, timeout=1.0)
        if alive:
            for p in alive:
                print(f"{p} survived SIGKILL, please contact the user: "
                      f"{p.username()}")
        else:
            print("Residual processes have been terminated!!!")


class FAI:
    def __init__(self):

        self._gui = None

        # Redis server must be started at first since when the GUI starts,
        # it needs to write all the configuration into Redis.
        try:
            start_redis_server()
        except (ConnectionError, FileNotFoundError):
            sys.exit(1)

        try:
            # process which runs the image assembler and processor
            self.image_worker = ImageWorker()
            # process which runs the scheduler
            self.scheduler = Scheduler()
            self.scheduler.connectInputToOutput(self.image_worker.output)

            ImageToolWindow.reset()

            self._gui = MainGUI(start_thread_logger=True)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logger.error(repr(e))
            logger.error(repr((traceback.format_tb(exc_traceback))))
            sys.exit(1)

    def init(self):
        logger.info(f"Number of available CPUs: {_N_CPUS}, "
                    f"number of available GPUs: {_N_GPUS}, "
                    f"total system memory: {_SYS_MEMORY/1024**3:.1f} GB")
        self.image_worker.start()
        register_fai_process(ProcessInfo(name=self.image_worker.name,
                                         process=self.image_worker))
        self.scheduler.start()
        register_fai_process(ProcessInfo(name=self.scheduler.name,
                                         process=self.scheduler))

        self._gui.connectInputToOutput(self.scheduler.output)
        self._gui.start_sgn.connect(self.start)
        self._gui.stop_sgn.connect(self.pause)

        return self

    def start(self):
        self.scheduler.resume()
        self.image_worker.resume()

    def pause(self):
        self.scheduler.pause()
        self.image_worker.pause()

    def terminate(self):
        if self._gui is not None:
            self._gui.close()

    @property
    def gui(self):
        return self._gui


def _parse_detector_name(detector):
    if detector == 'JUNGFRAU':
        return 'JungFrau'

    if detector == 'FASTCCD':
        return 'FastCCD'

    if detector == 'BASLERCAMERA':
        return 'BaslerCamera'

    return detector.upper()


def application():
    parser = argparse.ArgumentParser(prog="karaboFAI")
    parser.add_argument('-V', '--version', action='version',
                        version="%(prog)s " + __version__)
    parser.add_argument("detector", help="detector name (case insensitive)",
                        choices=[det.upper() for det in config.detectors],
                        type=lambda s: s.upper())
    parser.add_argument('--debug', action='store_true',
                        help="Run in debug mode")
    parser.add_argument("--topic", help="Name of the instrument",
                        type=lambda s: s.upper(),
                        choices=['FXE', 'HED', 'MID', 'SCS', 'SPB', 'SQS'],
                        default='UNKNOWN')

    args = parser.parse_args()
    health_check()

    if args.debug:
        logger.debug("'faulthandler enabled")
    else:
        logger.setLevel("INFO")

    detector = _parse_detector_name(args.detector)
    topic = args.topic
    if not faulthandler.is_enabled():
        faulthandler.enable(all_threads=False)

    app = mkQApp()

    # update global configuration
    config.load(detector)
    config.set_topic(topic)

    fai = FAI().init()

    app.exec_()


def kill_application():
    """KIll the application processes in case the app was shutdown unusually.

    If there is SEGFAULT, the child processes won't be clean up.
    """
    logger.setLevel('CRITICAL')

    py_procs = []  # Python processes
    thirdparty_procs = []  # thirdparty processes like redis-server
    for proc in psutil.process_iter(attrs=['pid', 'cmdline']):
        cmdline = proc.info['cmdline']
        if cmdline and 'karaboFAI' in cmdline[0]:
            thirdparty_procs.append(proc)
        elif len(cmdline) > 2 and 'karaboFAI.services' in cmdline[2]:
            py_procs.append(proc)

    # kill Python processes first
    for proc in py_procs:
        proc.kill()
        print(f"Sent SIGKILL to {proc} ...")

    for proc in thirdparty_procs:
        proc.kill()
        print(f"Sent SIGKILL to {proc} ...")

    gone, alive = psutil.wait_procs(
        itertools.chain(py_procs, thirdparty_procs), timeout=1.0)

    if alive:
        for p in alive:
            print(f"{p} survived SIGKILL, "
                  f"please try again or clean it manually")


def stream_file():
    ap = argparse.ArgumentParser(prog="karaboFAI-stream")
    ap.add_argument("detector", help="detector name (case insensitive)",
                    choices=[det.upper() for det in config.detectors],
                    type=lambda s: s.upper())
    ap.add_argument("port", help="TCP port to run server on")

    args = ap.parse_args()

    detector = _parse_detector_name(args.detector)

    app = mkQApp()
    streamer = FileStreamControllerWindow(detector=detector, port=args.port)
    app.exec_()


if __name__ == "__main__":

    application()
