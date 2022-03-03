import threading
import multiprocessing

from extra_data import DataCollection
from karabo_bridge import Client

from extra_foam.utils import get_available_port
from extra_foam.offline import run_info, gather_sources, serve_files


def test_run_info():
    assert run_info(None) == (0, -1, -1)

def test_gather_sources(mock_run_data):
    ret = gather_sources(DataCollection([]))
    assert ret == (dict(), dict(), dict())

    ret = gather_sources(mock_run_data)
    assert ret[0] == {
        "FXE_XAD_JF1M/DET/RECEIVER-1:daqOutput": ['data.adc', '*', 'data.gain', 'data.mask']
    }
    assert ret[1] == {
        "SA1_XTD2_XGM/DOOCS/MAIN:output":
            ['data.intensitySa1TD', 'data.intensitySa3TD', '*', 'data.xTD', 'data.yTD'],
        "SA1_XTD9_XGM/DOOCS/MAIN:output":
            ['data.intensitySa1TD', 'data.intensitySa3TD', '*', 'data.xTD', 'data.yTD'],
    }
    assert ret[2] == {
        "FXE_SMS_USR/MOTOR/UM01": ['actualPosition.value', '*', 'tolerance.value'],
        "FXE_SMS_USR/MOTOR/UM02": ['actualPosition.value', '*', 'tolerance.value'],
        'SA1_XTD9_XGM/DOOCS/MAIN':
            ['beamPosition.ixPos.value', 'beamPosition.iyPos.value', 'pulseEnergy.photonFlux.value',
             '*', 'hv.hamph.value']
    }

def test_serve_files(mock_run_data):
    port = get_available_port(12345)
    shared_tid = multiprocessing.Value("i")
    shared_rate = multiprocessing.Value("f")

    sources = []
    all_run_sources = mock_run_data.instrument_sources + mock_run_data.control_sources
    for source in all_run_sources:
        sources.extend([(source, key) for key in mock_run_data.keys_for_source(source)])

    server_thread = threading.Thread(target=serve_files,
                                     args=(mock_run_data, port, shared_tid, shared_rate, 50),
                                     kwargs={"sources": sources})
    server_thread.start()

    client = Client(f"tcp://127.0.0.1:{port}", timeout=1)
    for tid in mock_run_data.train_ids:
        data, meta = client.next()
        source = list(data.keys())[0]

        # Check the train ID
        assert meta[source]["timestamp.tid"] == tid

        # Check that all sources are present
        for source in sources:
            assert source[0] in data
            assert source[1] in data[source[0]]

    server_thread.join(timeout=5)
