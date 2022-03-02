from extra_data import DataCollection

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
