FXE tools for train integration
===============================

All the following requires Python 3.5 or greater

Server
------
To use this server launch it as follows::

    python server.py 4545 /path/to/run/dir /path/to/geometry_file.h5

The server reads the run and stores a certain quantity of trains in memory
before serving them on request of a KaraboBridge client.

.. note:
    The images are currently served with axes moved, as provided by the 
    online calibration pipeline.

Receiving Data
--------------
The data is received using the KaraboBridge interface::

    import matplotlib.pyplot as plt
    from karabo_bridge import KaraboBridge

    KaraboBridge("tcp://localhost:4545")
    data = client.next()
    d = data["FXE_DET_LPD1M-1/DET/combined"]["image.data"]
    
    plt.imshow(d[0], vmin=-10, vmax=6000)
    plt.show()  # Will display data with geometry
