import h5py
import numpy as np
import matplotlib.pyplot as plt


def readh5(filename, trainId, pulseId):
    """Can be done more effectively by using h5 index
    rather than converting whole data set into array"""
    f = h5py.File(filename, 'r')
    # compute index from trainID and pulseID
    index = trainId*0 + pulseId  # FIX ME

    print("index = {}".format(index))
    I_corr = np.array(f['DATA/I_CORRECTED'])
    I = np.array(f['DATA/I_UNCORRECTED'])
    q = np.array(f['DATA/Q'])
    # print(q.shape , I.shape, I_corr.shape)
    return q[index,:], I[index,:], I_corr[index,:]

def plot(q, I, I_corr, title):
    plt.plot(q, I, label="I_uncorrected")
    plt.plot(q, I_corr, label="I_corrected")
    plt.legend()
    plt.ylabel("Intensity")
    plt.xlabel("q")
    plt.title(title)

def save(q, I, I_corr, basename, header):
    filename = basename + '-I.txt'
    np.savetxt(filename, I,
               header=header)
    print("Written file {}".format(filename))

    filename = basename + '-I_corr.txt'
    np.savetxt(filename, I_corr,
               header=header)
    print("Written file {}".format(filename))

    filename = basename + '-q.txt'
    np.savetxt(filename, q,
               header=header)
    print("Written file {}".format(filename))


def read(filename):
    return np.loadtxt(filename)
