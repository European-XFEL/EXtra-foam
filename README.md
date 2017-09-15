Azimuthal integration snippets, to be integrated into https://github.com/European-XFEL/h5tools-py
eventually



## Example
<pre>

(azimuthal-integration) bash-3.2$ git checkout tags/v0.1
M	digging-in-output-file.ipynb
Note: checking out 'tags/v0.1'.

(azimuthal-integration) bash-3.2$ python toolA.py ~/Desktop/data/r0079
Unable to import pyOpenCl. Please install it from: http://pypi.python.org/pypi/pyopencl
Output file name is INT-R0079-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD00-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD01-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD02-S00000.h5
Couldn't open /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD03-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD04-S00000.h5
Couldn't open /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD05-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD06-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD07-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD08-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD09-S00000.h5
Couldn't open /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD10-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD11-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD12-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD13-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD14-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD15-S00000.h5
Full Image Set #1; Pulse #1 prepared in 170 ms
Azimuthal integration of image took 452  ms
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD00-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD01-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD02-S00000.h5
Couldn't open /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD03-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD04-S00000.h5
Couldn't open /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD05-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD06-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD07-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD08-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD09-S00000.h5
Couldn't open /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD10-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD11-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD12-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD13-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD14-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD15-S00000.h5
Full Image Set #1; Pulse #2 prepared in 151 ms
Azimuthal integration of image took 69  ms
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD00-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD01-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD02-S00000.h5
Couldn't open /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD03-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD04-S00000.h5
Couldn't open /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD05-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD06-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD07-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD08-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD09-S00000.h5
Couldn't open /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD10-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD11-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD12-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD13-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD14-S00000.h5
Opened file /Users/fangohr/Desktop/data/r0079/RAW-R0079-LPD15-S00000.h5
Full Image Set #1; Pulse #3 prepared in 141 ms
Azimuthal integration of image took 61  ms
(azimuthal-integration) bash-3.2$ python toolB.py INT-R0079-S00000.h5 0 2 histoffline
h5file = INT-R0079-S00000.h5
trainId = 0
pulseId = 2
outputfilename = histoffline
index = 2
(512,) (512,) (512,)
Written file histoffline-I.txt
Written file histoffline-I_corr.txt
Written file histoffline-q.txt
(azimuthal-integration) bash-3.2$ python toolC.py histoffline-I_corr.txt INT-R0079-S00000.h5 0 2
offlinefile = histoffline-I_corr.txt
h5file = INT-R0079-S00000.h5
trainId = 0
pulseId = 2
index = 2
(512,) (512,) (512,)
(azimuthal-integration) bash-3.2$
</pre>
