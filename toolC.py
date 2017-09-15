"""Tool to read histogram from h5 file, substract a (offline) histogram,
and display.

Example usage:

$> python toolC.py offlinedata-I.txt INT-R0079-S00000.h5 0 2

"""

import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt

from lib import readh5, save, plot, read


def main(offlinefile, h5file, trainId, pulseId):
    print("offlinefile = {}".format(offlinefile))
    print("h5file = {}".format(h5file))
    print("trainId = {}".format(trainId))
    print("pulseId = {}".format(pulseId))
    q, I, I_corr = readh5(h5file, trainId, pulseId)
    print (q.shape, I.shape, I_corr.shape)

    offline = read(offlinefile)
    I_ = I - offline
    I_corr_ = I_corr - offline

    summary = "h5file={}, trainId={}, pulseID={}, offline={}".format(
             h5file, trainId, pulseId, offlinefile)

    #save(q, I, I_corr, outputfilename, header=summary)
    plot(q, I_, I_corr_, summary)
    plt.show()




if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("""Usage
        toolb.py offlinefile h5file trainId pulseId""")
        sys.exit(1)

    offlinefile = sys.argv[1]
    h5file = sys.argv[2]
    trainId = int(sys.argv[3])
    pulseId = int(sys.argv[4])
    if len(sys.argv) > 5:
        raise ValueError("Only need 4 arguments.")

    main(offlinefile, h5file, trainId, pulseId)
