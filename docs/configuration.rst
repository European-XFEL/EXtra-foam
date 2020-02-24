.. code-block:: yaml

    DSSC: # detector name
        # <Karabo device ID> + space + <property>
        FXE_DET_LPD1M-1/CAL/APPEND_CORRECTED image.data
        FXE_DET_LPD1M-1/DET/*CH0:xtdf image.data
        # absolute path of geometry file
        GEOMETRY_FILE: geometries/lpd_mar_18_axesfixed.h5
        # quadrant coordinates for assembling detector modules
        QUAD_POSITIONS:
          x1:   11.4
          y1:  299
          x2:  -11.5
          y2:    8
          x3:  254.5
          y3:  -16
          x4:  278.5
          y4:  275
        # TCP address of the ZMQ bridge
        BRIDGE_ADDR": 10.253.0.53
        # TCP port of the ZMQ bridge
        BRIDGE_PORT": 4501
        # TCP address of the file streamer
        LOCAL_ADDR: 127.0.0.1
        # TCP port of the file streamer
        LOCAL_PORT: 45454
        # distance from sample to detector plan (orthogonal distance,
        # not along the beam), in meter
        SAMPLE_DISTANCE": 0.4
        # photon energy, in keV
        PHOTON_ENERGY": 9.3

.. code-block:: yaml

    XGM:
        # For control data: <Karabo device ID> <property>
        SA1_XTD2_XGM/DOOCS/MAIN pulseEnergy.photonFlux
        SA1_XTD2_XGM/DOOCS/MAIN beamPosition.ixPos
        SA1_XTD2_XGM/DOOCS/MAIN beamPosition.iyPos
        # For pipeline data: <Karabo device ID>:<output channel>
        SA1_XTD2_XGM/DOOCS/MAIN:output:
            property:
                - data.intensityTD
                - data.intensitySa1TD
                - data.intensitySa3TD
            vrange:
                - 0.0
                - .inf

.. code-block:: yaml

    Motor: # must be Motor
        # <Karabo device ID> + space + <property>
        FXE_SMS_USR/MOTOR/UM01 actualPosition
        FXE_SMS_USR/MOTOR/UM02 actualPosition
        FXE_SMS_USR/MOTOR/UM04 actualPosition:
            vrange:
                - 0.0
                - .inf