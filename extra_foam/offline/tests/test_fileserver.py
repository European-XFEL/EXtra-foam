from unittest.mock import MagicMock

from extra_data import DataCollection

from extra_foam.offline import (
    run_info, gather_sources
)


class TestFileServer:
    def testRunInfo(self):
        assert run_info(None) == (0, -1, -1)

    def testGatherSources(self):
        ret = gather_sources(DataCollection([]))
        assert ret == (dict(), dict(), dict())

        # test rd_raw is None

        rd = MagicMock()
        rd.instrument_sources = [
            "FXE_XAD_JF1M/DET/RECEIVER-1:daqOutput",
            "SA1_XTD2_XGM/DOOCS/MAIN:output",
            "SA1_XTD9_XGM/DOOCS/MAIN:output",
        ]
        rd.control_sources = [
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

        rd.keys_for_source.side_effect = keys_for_source
        ret = gather_sources(rd)
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
