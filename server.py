from collections import deque
from functools import partial
import msgpack
import msgpack_numpy
import numpy as np
import pickle
import sys
from threading import Thread
from time import sleep, time
import zmq

from euxfel_h5tools import RunHandler, stack_detector_data
from lpd_tools import LPDConfiguration, stitch_image

msgpack_numpy.patch()


# LPD Definitions used for generated data
_PULSES = 32
_MODULES = 16
_MOD_X = 256
_MOD_Y = 256
_SHAPE = (_PULSES, _MODULES, _MOD_X, _MOD_Y)

config = LPDConfiguration(hole_size=-26.28e-3, q_offset=3)


def get_stacked_detector_data(path):
    """Get the data from the files in the directory, as a
        stack to be assembled later.
        :param string path: the path where the data lives
        :return: a dictionary that can be decoded by the
                KaraboBridge client.
    """
    handler = RunHandler(path)
    for tid, data in handler.trains():
        image_data = stack_detector_data(data, "image.data", only="LPD")
        # We squeeze the second dimension which is of size one. FOR NOW??
        image_data = image_data.squeeze()
        data['FXE_DET_LPD1M-1/DET/combined'] = {"image.data": image_data}
        yield tid, data


def get_assembled_detector_data(path):
    """Get the data from the files in the directory, assembled.
        :param string path: the path where the data lives
        :return: a dictionary that can be decoded by the
                KaraboBridge client.
    """
    handler = RunHandler(path)
    for tid, data in handler.trains():
        image_data = stack_detector_data(data, "image.data", only="LPD")
        # We squeeze the second dimension which is of size one. FOR NOW??
        image_data = image_data.squeeze()
        image_data = stitch_image(config, image_data)
        data['FXE_DET_LPD1M-1/DET/combined'] = {"image.data": image_data}
        yield tid, data


def gen_combined_detector_data(source):
    """Generates random image data, useful if you don't have
       a run and want to get a hang on the data format
       :param string source: the source of the data you'll find
            in the dictionary. Use that same source to find the
            data in your client.
       :return: a dictionary that can be decoded by the
            KaraboBridge client.
    """
    gen = {source: {}}

    # metadata
    sec, frac = str(time()).split('.')
    tid = int(sec+frac[:1])
    gen[source]['metadata'] = {
        'source': source,
        'timestamp': {'tid': tid, 'sec': int(sec), 'frac': int(frac)}
    }

    # detector random data
    rand_data = partial(np.random.uniform, low=1500, high=1600,
                        size=(_MOD_X, _MOD_Y))
    data = np.zeros(_SHAPE, dtype=np.uint16)  # np.float32
    for pulse in range(_PULSES):
        for module in range(_MODULES):
            data[pulse, module] = rand_data()
    cellId = np.array([i for i in range(_PULSES)], dtype=np.uint16)
    length = np.ones(_PULSES, dtype=np.uint32) * int(131072)
    pulseId = np.array([i for i in range(_PULSES)], dtype=np.uint64)
    trainId = np.ones(_PULSES, dtype=np.uint64) * int(tid)
    status = np.zeros(_PULSES, dtype=np.uint16)

    gen[source]['image.data'] = data
    gen[source]['image.cellId'] = cellId
    gen[source]['image.length'] = length
    gen[source]['image.pulseId'] = pulseId
    gen[source]['image.trainId'] = trainId
    gen[source]['image.status'] = status

    checksum = np.ones(16, dtype=np.int8)
    magicNumberEnd = np.ones(8, dtype=np.int8)
    status = 0
    trainId = tid

    gen[source]['trailer.checksum'] = checksum
    gen[source]['trailer.magicNumberEnd'] = magicNumberEnd
    gen[source]['trailer.status'] = status
    gen[source]['trailer.trainId'] = trainId

    data = np.ones(416, dtype=np.uint8)
    trainId = tid

    gen[source]['detector.data'] = data
    gen[source]['detector.trainId'] = trainId

    dataId = 0
    linkId = np.iinfo(np.uint64).max
    magicNumberBegin = np.ones(8, dtype=np.int8)
    majorTrainFormatVersion = 2
    minorTrainFormatVersion = 1
    pulseCount = _PULSES
    reserved = np.ones(16, dtype=np.uint8)
    trainId = tid

    gen[source]['header.dataId'] = dataId
    gen[source]['header.linkId'] = linkId
    gen[source]['header.magicNumberBegin'] = magicNumberBegin
    gen[source]['header.majorTrainFormatVersion'] = majorTrainFormatVersion
    gen[source]['header.minorTrainFormatVersion'] = minorTrainFormatVersion
    gen[source]['header.pulseCount'] = pulseCount
    gen[source]['header.reserved'] = reserved
    gen[source]['header.trainId'] = tid

    return gen


def read_from_files(source, path, queue):
    # getter = get_stacked_detector_data(path)
    getter = get_assembled_detector_data(path)

    while True:
        if len(queue) < queue.maxlen:
            tid, data = next(getter)
            queue.append(data)
            print(tid)
        else:
            sleep(0.1)


def generate(source, queue):
    while True:
        if len(queue) < queue.maxlen:
            data = gen_combined_detector_data(source)
            queue.append(data)
            print(data[source]['metadata']['timestamp']['tid'])
        else:
            sleep(0.1)


def main(source, port, path, ser="msgpack"):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:{}'.format(port))

    if ser == 'msgpack':
        serialize = partial(msgpack.dumps, encoding='utf-8')
    elif ser == 'pickle':
        serialize = pickle.dumps

    queue = deque(maxlen=10)

    t = Thread(target=read_from_files, args=(source, path, queue,))
    t.start()

    while True:
        msg = socket.recv()
        if msg == b'next':
            while len(queue) <= 0:
                sleep(0.1)
            socket.send(serialize(queue.popleft()))
        else:
            print('wrong request')
            break
    else:
        socket.close()
        context.destroy()


if __name__ == '__main__':
    source = 'FXE_DET_LPD1M-1/DET/combined'
    if len(sys.argv) < 3:
        print("Usage:\n\tpython {} PORT PATH".format(sys.argv[0]))
        exit()
    _, port, path = sys.argv
    main(source, port, path)
