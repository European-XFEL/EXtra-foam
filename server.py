from asyncio import (
    FIRST_COMPLETED, Future, get_event_loop, Lock, sleep, wait, wait_for
)
from collections import deque
from functools import partial
import logging
import msgpack
import msgpack_numpy
import numpy as np
import sys
import zmq

from h5py import File

from karabo_data import RunDirectory, RunHandler, stack_detector_data
from karabo_data.geometry import LPDGeometry

from constants import bridge_key

msgpack_numpy.patch()

logging.basicConfig(level=logging.INFO,
                    format="PID %(process)5s %(name)18s: %(message)s",
                    stream=sys.stderr)
log = logging.getLogger()


class DataServer(Future):
    def __init__(self, port, datapath, geopath, loop=None):
        super().__init__(loop=loop)
        self.q_lock = Lock()
        self.port = port
        self.path = datapath
        self.queue = deque(maxlen=10)
        self.loop = loop

        self.serialize = partial(msgpack.dumps,
                                 use_bin_type=True,
                                 default=msgpack_numpy.encode)

        # The following is valid for the 20180318 geometry
        pos = [(-11.4, -229), (11.5, -8), (-254.5, 16), (-278.5, -275)]
        with File(geopath, 'r') as f:
            self.geometry = LPDGeometry.from_h5_file_and_quad_positions(f, pos)

        log.info("%s instantiated" % self.__class__)

    def get_detector_data(self):
        """Get the data from the files in the directory, assembled.
            :param string path: the path where the data lives
            :return: a dictionary that can be decoded by the
                    KaraboBridge client.
        """
        run = RunHandler(self.path)
        log.info("Run is %s" % run)
        for tid, data in run.trains():
            image_data = stack_detector_data(data, "image.data", only="LPD")
            cell_data = stack_detector_data(data, "image.cellId", only="LPD")
            # We squeeze the second dimension which is of size one. FOR NOW??
            image_data = image_data.squeeze()
            image_data, centre = self.geometry.position_all_modules(image_data)

            # Fake being the online preview with funny axes
            image_data = np.moveaxis(image_data, 0, 2)
            image_data[np.where(image_data == np.nan)] = 0

            print(image_data.shape)
            data[bridge_key] = {"image.data": image_data,
                                "image.cellId": cell_data,
                                "header.trainId": tid}
            yield tid, data

    async def blocking(self, func, *args):
        return await self.loop.run_in_executor(None, func, *args)

    async def read_files(self):
        log.info("Starting reading files")
        getter = self.get_detector_data()

        while True:
            while not len(self.queue) < self.queue.maxlen:
                await sleep(0.1)
            try:
                log.info("getting")
                tid, data = await self.blocking(next, getter)
                log.info("appended")
                self.queue.append(data)
                log.info(tid)
            except StopIteration:
                break

        self.set_result("Served all trains")
        return

    async def serve(self):
        log.info("Starting serving data")
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind('tcp://*:%s' % self.port)

        while True:
            msg = await self.blocking(socket.recv)
            if msg == b'next':
                with await self.q_lock:
                    data = self.queue.popleft()
                    await self.blocking(socket.send, self.serialize(data))
            else:
                log.error("request was %s" % msg)
                break
            await sleep(0)  # Give the event loop a chance
        else:
            await self.blocking(socket.close)
            await self.blocking(context.destroy)


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

    loop.run_until_complete(wait(tasks, return_when=FIRST_COMPLETED))
    loop.close()
