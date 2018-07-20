import time

from data_processing import process_data
from logger import log


def acquire_data(client):
    # retrieve
    t0 = time.perf_counter()
    kb_data = client.next()
    log.info("Time for retrieving data from the server: {:.1f} ms"
             .format(1000 * (time.perf_counter() - t0)))

    # process
    t0 = time.perf_counter()
    data = process_data(kb_data)
    log.info("Time for processing the data: {:.1f} ms"
             .format(1000 * (time.perf_counter() - t0)))

    return data
