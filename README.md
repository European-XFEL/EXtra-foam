


Usage
=====
All the following requires Python 3

Server
------

To use this server launch it as follows:

```python
python server.py 4545 /path/to/run/dir
```
and connect your Karabo Bridge client to port 4545
The server can serve the data either as stacked frames or as assembled
images, just as one would from the output of the calibration pipeline.


Receiving Data
--------------
The data is received using the KaraboBridge interface:

```python
    import matplotlib.pyplot as plt
    from karabo_bridge import KaraboBridge

    KaraboBridge("tcp://localhost:4545")
    data = client.next()
    d = data["FXE_DET_LPD1M-1/DET/combined"]["image.data"]
    
    plt.imshow(d[0], vmin=-10, vmax=6000)
    plt.show()  # will display an assembled image, without offsets
```

There are a few utilities to checkout the data.

[plot2d](plotUpdateFromThread2d.py) will plot the the first pulse's image as received.
[plot1d](plotUpdateFromThread2d.py) will perform the integration found in [integration](integration.py) and display the result live for all pulses in each train.

[plotUpdate](plotUpdate.py) will show both side by side as follows:
![screenshot of plotUpdate](plotUpdate.png)

Documentation
=============
The [LDP Images from Bridge notebook](HowTo.ipynb) demonstrates
how to manipulate the data coming from the bridge, as processed
by the [lpd_tools](lpd_tools.py).
