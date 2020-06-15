"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import argparse
import os
import datetime
import time
import faulthandler
import sys
import traceback
import itertools
import multiprocessing as mp

import psutil

import redis

from . import __version__
from .config import AnalysisType, config, PipelineSlowPolicy
from .database import Metadata as mt
from .ipc import init_redis_connection
from .logger import logger
from .gui import MainGUI, mkQApp
from .pipeline import PulseWorker, TrainWorker
from .processes import register_foam_process
from .utils import check_system_resource, query_yes_no
from .gui.windows import FileStreamWindow

_CPU_INFO, _GPU_INFO, _MEMORY_INFO = check_system_resource()


def try_to_connect_redis_server(host, port, *, password=None, n_attempts=5):
    """Try to connect to a starting Redis server.

    :param str host: IP address of the Redis server.
    :param int port:: Port of the Redis server.
    :param str password: password for the Redis server.
    :param int n_attempts: Number of attempts to connect to the redis server.

    :return: Redis connection client.

    Raises:
        ConnectionError: raised if the Redis server cannot be connected.
    """
    client = redis.Redis(host=host, port=port, password=password)

    for i in range(n_attempts):
        try:
            logger.info(f"Say hello to Redis server at {host}:{port}")
            client.ping()
        except (redis.ConnectionError, redis.InvalidResponse):
            time.sleep(1)
            logger.info("No response from the Redis server")
        else:
            logger.info("Received response from the Redis server")
            return client

    raise ConnectionError(f"Failed to connect to the Redis server at "
                          f"{host}:{port}.")


def start_redis_client():
    """Start a Redis client.

    This function is for the command line tool: extra-foam-redis-cli.
    """
    exec_ = config["REDIS_EXECUTABLE"].replace("redis-server", "redis-cli")

    if not os.path.isfile(exec_):
        raise FileNotFoundError(f"Redis executable file {exec_} not found!")

    command = [exec_]
    command.extend(sys.argv[1:])

    logger.setLevel("INFO")

    proc = psutil.Popen(command)
    proc.wait()


def start_redis_server(host='127.0.0.1', port=6379, *, password=None):
    """Start a Redis server.

    :param str host: IP address of the Redis server.
    :param int port:: Port of the Redis server.
    :param str password: password for the Redis server.
    """
    executable = config["REDIS_EXECUTABLE"]
    if not os.path.isfile(executable):
        logger.error(f"Unable to find the Redis executable file: "
                     f"{executable}!")
        sys.exit(1)

    # Construct the command to start the Redis server.
    # TODO: Add log rotation or something else which prevent logfile bloating
    command = [executable,
               "--port", str(port),
               "--loglevel", "warning",
               "--logfile", config["REDIS_LOGFILE"]]
    if password is not None:
        command.extend(["--requirepass", password])

    process = psutil.Popen(command)

    try:
        # wait for the Redis server to start
        try_to_connect_redis_server(host, port, password=password)
    except ConnectionError:
        # TODO: whether we need a back-up port for each detector?
        # Allow users to assign the port by themselves is also a disaster!
        logger.error(f"Unable to start a Redis server at {host}:{port}. "
                     f"Please check whether the port is already taken up.")
        sys.exit(1)

    if process.poll() is None:
        client = init_redis_connection(host, port, password=password)

        # Put a time stamp in Redis to indicate when it was started.
        client.hset(mt.SESSION, mapping={
            'detector': config["DETECTOR"],
            'topic': config["TOPIC"],
            'redis_server_start_time': time.time(),
        })

        # TODO: find a better place to do the initialization
        # Prevent 'has_analysis', 'has_any_analysis' and
        # 'has_all_analysis' from getting None when querying.
        client.hset(mt.ANALYSIS_TYPE, mapping={t: 0 for t in AnalysisType})

        logger.info(f"Redis server started at {host}:{port}")

        register_foam_process("redis", process)

        # subscribe List commands
        # client.config_set("notify-keyspace-events", "Kl")

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
            sys.exit(1)

        # Increase the hard and soft limits for the redis client pubsub buffer
        # to 512MB and 128MB, respectively.
        cli_buffer_cfg = (client.config_get("client-output-buffer-limit")[
            "client-output-buffer-limit"]).split()
        assert len(cli_buffer_cfg) == 12
        soft_limit = 128 * 1024 ** 2  # 128 MB
        soft_second = 60  # in second
        hard_limit = 4 * soft_limit
        cli_buffer_cfg[8:] = [
            "pubsub", str(hard_limit), str(soft_limit), str(soft_second)
        ]
        client.config_set("client-output-buffer-limit",
                          " ".join(cli_buffer_cfg))

    else:
        # It is unlikely to happen on the online cluster since we have
        # checked the existing Redis server before trying to start a new one.
        # Nevertheless, it could happen if someone just started a Redis
        # server after the check.
        logger.error(f"Unable to start a Redis server at {host}:{port}. "
                     f"Please check whether the port is already taken up.")
        sys.exit(1)


def check_existing_redis_server(host, port, password):
    """Check existing Redis servers.

    Allow to shut down the Redis server when possible and agreed.
    """
    def wait_after_shutdown(n):
        for _count in range(n):
            print(f"Start new Redis server after {n} seconds "
                  + "." * _count, end="\r")
            time.sleep(1)

    while True:
        try:
            client = try_to_connect_redis_server(
                host, port, password=password, n_attempts=1)
        except ConnectionError:
            break

        sess = client.hgetall(mt.SESSION)
        det = sess.get(b'detector', b'').decode("utf-8")
        start_time = sess.get(
            b'redis_server_start_time', b'').decode("utf-8")
        if start_time:
            start_time = datetime.datetime.fromtimestamp(
                float(start_time))

        if det == config["DETECTOR"]:
            logger.warning(
                f"Found Redis server for {det} (started at "
                f"{start_time}) already running on this machine "
                f"using port {port}!")

            if query_yes_no(
                    "\nYou can choose to shut down the Redis server. Please "
                    "note that the owner of the Redis server will be "
                    "informed (your username and IP address).\n\n"
                    "Shut down the existing Redis server?"
            ):
                try:
                    proc = psutil.Process()
                    killer = proc.username()
                    killer_from = proc.connections()
                    client.publish("log:warning",
                                   f"<{killer}> from <{killer_from}> "
                                   f"will shut down the Redis server "
                                   f"immediately!")
                    client.execute_command("SHUTDOWN")

                except redis.exceptions.ConnectionError:
                    logger.info("The old Redis server was shut down!")

                # ms -> s, give enough margin
                wait_time = int(config["REDIS_PING_ATTEMPT_INTERVAL"] / 1000
                                * config["REDIS_MAX_PING_ATTEMPTS"] * 2)
                wait_after_shutdown(wait_time)
                continue

            else:
                # not shutdown the existing Redis server
                sys.exit(0)

        else:
            logger.warning(f"Found Unknown Redis server running on "
                           f"this machine using port {port}!")
            # Unlike to happen, who sets a Redis server with the same
            # password??? Then we can boldly shut it down.
            try:
                client.execute_command("SHUTDOWN")
            except redis.exceptions.ConnectionError:
                logger.info("The unknown Redis server was shut down!")

            # give some time for any affiliated processes
            wait_after_shutdown(5)
            continue


class Foam:
    def __init__(self, redis_address="127.0.0.1"):

        self._gui = None

        host = redis_address
        port = config["REDIS_PORT"]
        password = config["REDIS_PASSWORD"]

        check_existing_redis_server(host, port, password)

        # Redis server must be started at first since when the GUI starts,
        # it needs to write all the configuration into Redis.
        start_redis_server(host, port, password=password)

        try:
            # The following two events controls the pause and close
            # of all processes.
            self._pause_ev = mp.Event()
            self._close_ev = mp.Event()

            self.pulse_worker = PulseWorker(self._pause_ev, self._close_ev)
            self.train_worker = TrainWorker(self._pause_ev, self._close_ev)
            self.train_worker.input.connect(self.pulse_worker.output)

            self._gui = MainGUI(self._pause_ev, self._close_ev)
            self._gui.input.connect(self.train_worker.output)

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logger.error(repr(e))
            logger.error(repr((traceback.format_tb(exc_traceback))))
            sys.exit(1)

    def init(self):
        logger.info(f"{_CPU_INFO}, {_GPU_INFO}, {_MEMORY_INFO}")

        self._gui.start_sgn.connect(self._pause_ev.set)
        self._gui.stop_sgn.connect(self._pause_ev.clear)
        self._gui.start()

        self.pulse_worker.start()
        self.train_worker.start()

        return self

    def terminate(self):
        if self._gui is not None:
            self._gui.close()

    @property
    def gui(self):
        return self._gui


def application():
    parser = argparse.ArgumentParser(prog="extra-foam")
    parser.add_argument('-V', '--version',
                        action='version',
                        version="%(prog)s " + __version__)
    parser.add_argument("detector",
                        help="detector name (case insensitive)",
                        choices=[det.upper() for det in config.detectors]
                                + ["JUNGFRAUPR"],
                        type=lambda s: s.upper())
    parser.add_argument("topic",
                        help="name of the instrument",
                        choices=config.topics,
                        type=lambda s: s.upper())
    parser.add_argument("--n_modules",
                        help="Number of detector modules. It is only "
                             "available for using single-module detectors "
                             "like JungFrau in a combined way. Not all "
                             "single-module detectors are supported.",
                        default=None,
                        type=int)
    parser.add_argument('--debug',
                        action='store_true',
                        help="run in debug mode")
    parser.add_argument("--pipeline_slow_policy",
                        help="Pipeline policy when the processing rate is "
                             "slower than the arrival rate (0 for always "
                             "process the latest data and 1 for wait until "
                             "processing of the current data finishes).",
                        choices=[0, 1],
                        default=1,
                        type=int)
    parser.add_argument("--redis_address",
                        help="address of the Redis server",
                        default="127.0.0.1",
                        type=lambda s: s.lower())

    args = parser.parse_args()

    if args.debug:
        logger.setLevel("DEBUG")
    # No ideal whether it affects the performance. If it does, enable it only
    # in debug mode.
    faulthandler.enable(all_threads=False)

    detector = config.parse_detector_name(args.detector)
    topic = args.topic
    if detector == "JUNGFRAUPR":
        raise ValueError(f"Invalid detector type!\n\n"
                         f"Please use JungFrau instead and modify your config "
                         f"file {topic.lower()}.config.yaml accordingly: \n"
                         f"1. Move non-duplicated data sources under JungFrauPR "
                         f"into JungFrau;\n"
                         f"2. Remove JungFrauPR from 'SOURCE' and 'DETECTOR'.")

    redis_address = args.redis_address
    if redis_address not in ["localhost", "127.0.0.1"]:
        raise NotImplementedError("Connecting to remote Redis server is "
                                  "not supported yet!")

    app = mkQApp()
    app.setStyleSheet(
        "QTabWidget::pane { border: 0; }"
    )

    # update global configuration
    if detector in ("JungFrau", "ePix100"):
        n_modules = args.n_modules
        if n_modules is None:
            n_modules = 1

        config.load(detector, topic,
                    NUMBER_OF_MODULES=n_modules,
                    REQUIRE_GEOMETRY=n_modules > 1,
                    PIPELINE_SLOW_POLICY=PipelineSlowPolicy(args.pipeline_slow_policy))
    else:
        # TODO: consider moving the following check into "config.load"
        if args.n_modules is not None:
            raise ValueError(f"Number of modules of {detector} is not "
                             f"configurable!")
        config.load(detector, topic,
                    PIPELINE_SLOW_POLICY=PipelineSlowPolicy(args.pipeline_slow_policy))

    foam = Foam(redis_address=redis_address).init()

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
        if cmdline and config["REDIS_EXECUTABLE"] in cmdline[0]:
            thirdparty_procs.append(proc)
        elif len(cmdline) > 2 and 'extra_foam.services' in cmdline[2]:
            py_procs.append(proc)

    if not py_procs and not thirdparty_procs:
        print("Found no EXtra-foam process!")
        return

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
                  f"please try again or kill it manually")
    else:
        print("All the above EXtra-foam processes have been killed!")


def stream_file():
    parser = argparse.ArgumentParser(prog="extra-foam-stream")
    parser.add_argument("--port",
                        help="TCP port to run server on",
                        default="45454")

    args = parser.parse_args()

    app = mkQApp()
    streamer = FileStreamWindow(port=args.port)
    app.exec_()


if __name__ == "__main__":

    application()
