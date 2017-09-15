"""Tool to read histogram from h5 file, plot the histogram, and save
to file (so it can be used for computing the difference later)

Example usage:

$> python toolB.py INT-R0079-S00000.h5 0 2 histoffline
h5file = INT-R0079-S00000.h5
trainId = 0
pulseId = 2
outputfilename = histoffline
index = 2
(512,) (512,) (512,)
Written file histoffline-I.txt
Written file histoffline-I_corr.txt
Written file histoffline-q.txt
[matplotlib figure is displayed]
"""

import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt


from lib import readh5, plot, save

def main(h5file, trainId, pulseId, outputfilename):
    print("h5file = {}".format(h5file))
    print("trainId = {}".format(trainId))
    print("pulseId = {}".format(pulseId))
    print("outputfilename = {}".format(outputfilename))
    q, I, I_corr = readh5(h5file, trainId, pulseId)
    print (q.shape, I.shape, I_corr.shape)

    summary = "h5file={}, trainId={}, pulseID={}".format(
             h5file, trainId, pulseId)

    save(q, I, I_corr, outputfilename, header=summary)
    plot(q, I, I_corr, summary)
    plt.show()




if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("""Usage
        toolb.py h5file trainId pulseId [outputfilename]""")
        sys.exit(1)

    outputfilename = "toolb-output-base"
    h5file = sys.argv[1]
    trainId = int(sys.argv[2])
    pulseId = int(sys.argv[3])
    if len(sys.argv) > 4:
        outputfilename = sys.argv[4]
    if len(sys.argv) > 5:
        raise ValueError("Only need 4 arguments.")

    main(h5file, trainId, pulseId, outputfilename)
