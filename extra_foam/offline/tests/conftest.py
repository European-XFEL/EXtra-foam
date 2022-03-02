from unittest.mock import MagicMock

import pytest

@pytest.fixture
def mock_run_data():
    run = MagicMock()
    run.instrument_sources = [
        "FXE_XAD_JF1M/DET/RECEIVER-1:daqOutput",
        "SA1_XTD2_XGM/DOOCS/MAIN:output",
        "SA1_XTD9_XGM/DOOCS/MAIN:output",
    ]
    run.control_sources = [
        "FXE_SMS_USR/MOTOR/UM01",
        "FXE_SMS_USR/MOTOR/UM02",
        "SA1_XTD9_XGM/DOOCS/MAIN"
    ]

    def keys_for_source(src):
        if "MOTOR" in src:
            return {
                "actualPosition.value", "actualPosition.timestamp",
                "tolerance.value", "tolerance.timestamp",
            }
        elif "DET" in src:
            return {
                "data.adc", "data.mask", "data.gain"
            }
        elif "XGM" in src:
            if "output" in src:
                return {
                    "data.intensitySa1TD",
                    "data.intensitySa3TD",
                    "data.xTD",
                    "data.yTD"
                }
            else:
                return {
                    "pulseEnergy.photonFlux.value",
                    "pulseEnergy.photonFlux.timestamp",
                    "beamPosition.ixPos.value",
                    "beamPosition.ixPos.timestamp",
                    "beamPosition.iyPos.value",
                    "beamPosition.iyPos.timestamp",
                    "hv.hamph.value",
                    "hv.hamph.timestamp"
                }
        else:
            return {
                "tolerance.value", "tolerance.timestamp",
            }

    run.keys_for_source.side_effect = keys_for_source

    yield run
