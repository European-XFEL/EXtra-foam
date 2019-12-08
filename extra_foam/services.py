"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

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
from .database import MetaProxy
from .ipc import redis_connection, reset_redis_connections
from .logger import logger
from .gui import MainGUI, mkQApp
from .pipeline import PulseWorker, TrainWorker
from .processes import ProcessInfo, register_foam_process
from .utils import check_system_resource, query_yes_no
from .gui.windows import FileStreamControllerWindow

_CPU_INFO, _GPU_INFO, _MEMORY_INFO = check_system_resource()


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
    """Start a Redis client.

    This function is for the command line tool: extra-foam-redis-cli.
    """
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

        register_foam_process(ProcessInfo(name="redis", process=process))

        try:
            frac = config["REDIS_MAX_MEMORY_FRAC"]
            if frac < 0.01 or frac > 0.5:
                frac = 0.3  # in case of evil configuration
            client.config_set("maxmemory", int(frac*_MEMORY_INFO.total_memory))
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
                'extra_foam.services' in proc.info['cmdline']:
            residual.append(proc)

    if residual:
        if query_yes_no(
            "Warning: Found old extra-foam instance(s) running in this "
            "machine!!!\n\n"
            "Running more than two extra-foam instances with the same \n"
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


class Foam:
    def __init__(self):

        self._gui = None

        # Redis server must be started at first since when the GUI starts,
        # it needs to write all the configuration into Redis.
        try:
            start_redis_server()
        except (ConnectionError, FileNotFoundError):
            sys.exit(1)

        proxy = MetaProxy()
        proxy.set_session({'detector': config['DETECTOR'],
                           'topic': config['TOPIC']})
        proxy.initialize_analysis_types()

        try:
            self.pulse_worker = PulseWorker()
            self.train_worker = TrainWorker()
            self.train_worker.connectInputToOutput(self.pulse_worker.output)

            self._gui = MainGUI(start_thread_logger=True)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logger.error(repr(e))
            logger.error(repr((traceback.format_tb(exc_traceback))))
            sys.exit(1)

    def init(self):
        logger.info(f"{_CPU_INFO}, {_GPU_INFO}, {_MEMORY_INFO}")
        self.pulse_worker.start()
        register_foam_process(ProcessInfo(name=self.pulse_worker.name,
                                          process=self.pulse_worker))
        self.train_worker.start()
        register_foam_process(ProcessInfo(name=self.train_worker.name,
                                          process=self.train_worker))

        self._gui.connectInputToOutput(self.train_worker.output)
        self._gui.start_sgn.connect(self.start)
        self._gui.stop_sgn.connect(self.pause)

        return self

    def start(self):
        self.train_worker.resume()
        self.pulse_worker.resume()

    def pause(self):
        self.train_worker.pause()
        self.pulse_worker.pause()

    def terminate(self):
        if self._gui is not None:
            self._gui.close()

    @property
    def gui(self):
        return self._gui


def application():
    parser = argparse.ArgumentParser(prog="extra-foam")
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

    detector = config.parse_detector_name(args.detector)
    topic = args.topic
    if not faulthandler.is_enabled():
        faulthandler.enable(all_threads=False)

    app = mkQApp()
    app.setStyleSheet(
        "QTabWidget::pane { border: 0; }"
    )

    # update global configuration
    config.load(detector)
    config.set_topic(topic)

    foam = Foam().init()

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
        if cmdline and 'extra-foam' in cmdline[0]:
            thirdparty_procs.append(proc)
        elif len(cmdline) > 2 and 'extra_foam.services' in cmdline[2]:
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
    ap = argparse.ArgumentParser(prog="extra-foam-stream")
    ap.add_argument("detector", help="detector name (case insensitive)",
                    choices=[det.upper() for det in config.detectors],
                    type=lambda s: s.upper())
    ap.add_argument("port", help="TCP port to run server on")

    args = ap.parse_args()

    detector = config.parse_detector_name(args.detector)

    app = mkQApp()
    streamer = FileStreamControllerWindow(detector=detector, port=args.port)
    app.exec_()


if __name__ == "__main__":

    application()
