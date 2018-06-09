from asyncio import FIRST_COMPLETED, get_event_loop, Lock, sleep, wait
from collections import deque
from functools import partial
import logging
from signal import SIGINT, SIGTERM
import sys
import time

from h5py import File
import msgpack
import msgpack_numpy
import numpy as np
import zmq
import zmq.asyncio as zio

from karabo_data import RunDirectory, stack_detector_data
from karabo_data.geometry import LPDGeometry

from constants import bridge_key

msgpack_numpy.patch()

logging.basicConfig(level=logging.INFO,
                    format="PID %(process)5s %(name)18s: %(message)s",
                    stream=sys.stderr)
log = logging.getLogger()


def cancel(tasks):
    for task in tasks:
        task.cancel()


class DataServer:
    def __init__(self, port, datapath, geopath, loop=None):
        self.q_lock = Lock()
        self.port = port
        self.path = datapath
        self.queue = deque(maxlen=10)
        self.loop = loop

        self.serialize = partial(msgpack.dumps,
                                 use_bin_type=True,
                                 default=msgpack_numpy.encode)

        # The following is valid-ish for the 20180318 geometry
        pos = [(-11.4, -229), (11.5, -8), (-254.5, 16), (-278.5, -275)]
        with File(geopath, 'r') as f:
            self.geometry = LPDGeometry.from_h5_file_and_quad_positions(f, pos)

    def get_detector_data(self):
        """Get the data from the files in the directory, assembled.
            :param string path: the path where the data lives
            :return: a dictionary that can be decoded by the
                    KaraboBridge client.
        """
        run = RunDirectory(self.path)
        run.info()
        for tid, data in run.trains():
            loadtime = time.time()
            image_data = stack_detector_data(data, "image.data", only="LPD")
            cell_data = stack_detector_data(data, "image.cellId", only="LPD")
            # We squeeze the second dimension which is of size one. FOR NOW??
            image_data = image_data.squeeze()
            image_data, centre = self.geometry.position_all_modules(image_data)

            # Fake being the online preview with funny axes
            image_data = np.moveaxis(image_data, 0, 2)

            # Replace the LPDGeometry-induced NaNs with zeroes
            image_data = np.nan_to_num(image_data)

            data[bridge_key] = {"image.data": image_data,
                                "image.cellId": cell_data,
                                "header.trainId": tid}
            yield tid, data, loadtime

    async def blocking(self, func, *args):
        return await self.loop.run_in_executor(None, func, *args)

    async def read_files(self):
        log.info("Starting reading files")
        getter = self.get_detector_data()

        while True:
            while not len(self.queue) < self.queue.maxlen:
                await sleep(0.1)
            with await self.q_lock:
                try:
                    tid, data, loadtime = await self.blocking(next, getter)
                    self.queue.append((tid, data, loadtime))
                    log.info("Loaded %s" % tid)
                except StopIteration:
                    break

        log.info("Served all trains")

    async def serve(self):
        log.info("Starting serving data")
        context = zio.Context()
        socket = context.socket(zmq.REP)
        socket.bind('tcp://*:%s' % self.port)

        while True:
            msg = await socket.recv()
            if msg == b'next':
                with await self.q_lock:
                    tid, data, loadtime = self.queue.popleft()
                    data = await self.blocking(self.serialize, data)
                    await socket.send(data)
                    log.info("Served %s" % tid)
                    time_ = time.time() - loadtime
                    log.debug("{} was in cached for {}".format(tid, time_))
            else:
                log.error("unsupported request %s" % msg)
                break
        else:
            await socket.close()
            await context.destroy()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        exit("Usage:\n\tpython %s PORT DATAPATH [GEOPATH]" % sys.argv[0])

    _, port, path = sys.argv[:3]

    geopath = "~/data/lpd_mar_18.h5"
    if len(sys.argv) > 3:
        geopath = sys.argv[3]

    loop = get_event_loop()
    server = DataServer(port, path, geopath, loop)

    tasks = [loop.create_task(server.read_files()),
             loop.create_task(server.serve())]
    loop.add_signal_handler(SIGINT, cancel, tasks)
    loop.add_signal_handler(SIGTERM, cancel, tasks)
    loop.run_until_complete(wait(tasks, return_when=FIRST_COMPLETED))
    loop.close()
