.. _config file:

CONFIG FILE
===========

Users from different instruments should use the corresponding config file to config
the instrument specific data sources, the detector specific setups and so on.

Each instrument has a default config file, which can be found in the
`github repository <https://github.com/European-XFEL/EXtra-foam/tree/dev/extra_foam/configs>`__.
We appreciate if the beamline scientists can help us keep the default config file updated.

Let's take *FXE* for example, when one starts a detector on topic *FXE* for the first by
time:

.. code-block:: bash

    extra-foam LPD FXE

, the system will create a new config file `$HOME/.EXtra-foam/fxe.config.yaml` using the default one.
The first block of the config file looks like the following:

.. code-block:: yaml

    SOURCE:
        # Default source type: FILES or BRIDGE
        DEFAULT_TYPE: 1
        CATEGORY:
            LPD:
                PIPELINE:
                    FXE_DET_LPD1M-1/CAL/APPEND_CORRECTED:
                        - image.data
                    FXE_DET_LPD1M-1/DET/*CH0:xtdf:
                        - image.data

            JungFrau:
                PIPELINE:
                    FXE_XAD_JF1M/DET/RECEIVER-1:display:
                        - data.adc
                    FXE_XAD_JF1M/DET/RECEIVER-2:display:
                        - data.adc

            XGM:
                CONTROL:
                    SA1_XTD2_XGM/DOOCS/MAIN:
                        - pulseEnergy.photonFlux
                        - beamPosition.ixPos
                        - beamPosition.iyPos
                PIPELINE:
                    SA1_XTD2_XGM/DOOCS/MAIN:output:
                        - data.intensitySa1TD

            Motor:
                CONTROL:
                    FXE_SMS_USR/MOTOR/UM01:
                        - actualPosition
                    FXE_SMS_USR/MOTOR/UM02:
                        - actualPosition


The next block of the config file looks like the following:

.. code-block:: yaml

    DETECTOR:
        LPD:
            GEOMETRY_FILE: lpd_mar_18_axesfixed.h5
            # - For lpd_mar_18.h5 and LPDGeometry in karabo_data
            # "QUAD_POSITIONS": [[-13.0, -299.0], [11.0, -8.0], [-254.0, 16.0], [-278.0, -275.0]],
            # - For lpd_mar18_axesfixed.h5 and LPD_1MGeometry in karabo_data
            # The geometry uses XFEL standard coordinate directions.
            QUAD_POSITIONS:
                x1:   11.4
                y1:  299
                x2:  -11.5
                y2:    8
                x3:  254.5
                y3:  -16
                x4:  278.5
                y4:  275
            BRIDGE_ADDR: 10.253.0.53
            BRIDGE_PORT: 4501
            LOCAL_ADDR: 127.0.0.1
            LOCAL_PORT: 45451
            SAMPLE_DISTANCE": 0.4
            PHOTON_ENERGY": 9.3

        JungFrau:
            BRIDGE_ADDR: 10.253.0.53
            BRIDGE_PORT: 4501
            LOCAL_ADDR: 127.0.0.1
            LOCAL_PORT: 45453
            SAMPLE_DISTANCE: 2.0
            PHOTON_ENERGY: 9.3
