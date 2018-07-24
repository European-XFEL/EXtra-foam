#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Aug 23 21:33:23 2017

@author: khakhulin
"""
#import sys

import h5py
#import os
import sys 
#import fabio
import numpy as np
#from detect_peaks import detect_peaks
#from scipy import optimize
#import scipy
#import pylab
#import pyopencl
import pyFAI, time, math
import matplotlib.pyplot as plt
#from IPython.display import display, Image
#from numpy import *
#import scipy.misc
#from PIL import Image

##%%
from karabo_data import RunDirectory, stack_detector_data
from karabo_data.geometry import LPDGeometry


#%
def h5open(fname):
    try:
        h5_file = h5py.File(fname, 'r')
    except:
        h5_file = 0
    return h5_file

def h5close(h5_file):
    h5py.File.close(h5_file)

def clip(array, min=-10000, max=100000):
    x = array.copy()
    finite = np.isfinite(x)
    # Suppress warnings comparing numbers to nan
    with np.errstate(invalid='ignore'):
        x[finite & (x < min)] = np.nan
        x[finite & (x > max)] = np.nan
    return x

pi = math.pi;
#dist = 150*1e-3 # m #2050
dist = 200*1e-3 # m # 2072

#center_X = 535
#center_Y = 530
center_Y = 620
center_X = 580

pixel_size = 0.5e-3
energy = 9.30 # keV
wavelength_lambda = 12.3984/energy*1e-10 # m

#muSi = 91 # silicon sensor absorption coefficient
#tSi = 500*1e-4 # sensor thickness

#mus = 6.3 # sample absorption coefficient
#ts = 100*1e-4 # sample thickness

#HoleSize = -26.28e-3
#HoleSize = -24.5e-3



#
#HoleSize = -26.25e-3
#HoleSize_pixels = np.abs(np.int(np.ceil(HoleSize/pixel_size)))
#
#Q_offset = 3; ################################### offset between quadrants
#
#SM_unit = 256
#dx_map = [0, 0, SM_unit, SM_unit, 0, 0, SM_unit, SM_unit, SM_unit*2, SM_unit*2, 
#          SM_unit*3, SM_unit*3, SM_unit*2, SM_unit*2, SM_unit*3, SM_unit*3,]
#dy_map = [0, SM_unit, SM_unit, 0, SM_unit*2, SM_unit*3, SM_unit*3, SM_unit*2,
#          SM_unit*2, SM_unit*3, SM_unit*3, SM_unit*2, 0, SM_unit, SM_unit, 0]
#
#FullIm = np.zeros([SM_unit*4,SM_unit*4],dtype='int16')
#CombFullIm = np.zeros([SM_unit*4+HoleSize_pixels+Q_offset,SM_unit*4+HoleSize_pixels+Q_offset],dtype='int16')
#TotalIm = np.zeros([SM_unit*4+HoleSize_pixels+Q_offset,SM_unit*4+HoleSize_pixels+Q_offset],dtype='int32')
# setting the integrator

RadialRange = (0.2,5)
BinSize = 0.02
#npt = int(((RadialRange[1]+BinSize/2)-(RadialRange[0]-BinSize/2))/BinSize/3)
npt = 512

ai = pyFAI.AzimuthalIntegrator(dist=dist,
                   poni1=center_Y*pixel_size,
                   poni2=center_X*pixel_size,
                   pixel1=pixel_size,
                   pixel2=pixel_size,
                   rot1=0,rot2=0,rot3=0,
                   wavelength=wavelength_lambda)

# q normalization range
Qnorm_min = 0.5
Qnorm_max = 4
#
#runNum = int(sys.argv[1])
#takeTrains=int(sys.argv[2])
#minSet = int(sys.argv[3])
#maxSet = int(sys.argv[4])
#plotfinal = int(sys.argv[5]) 
#saving = int(sys.argv[6])

#
runNum = 78
takeTrainsBegin =24 #54
takeTrainsEnd=34 #64
minSet  = 0
maxSet  = 0
plotfinal = 0 
saving = 0 
#
startTime = time.time()
#runNum = 298 # 565kHz veto range(0,120,8)
#runNum = 297 # 1.13MHz veto range(0,120,4)

runStr = 'r%04i/' %runNum
plotting = 1


quadpos = [(-11.4, -299), (11.5, -8), (-254.5, 16), (-278.5, -275)]  # MAR 18
# geometry_file = '/home/khakhuli/AzimuthalIntegration/lpd_mar_18.h5'
geometry_file = './lpd_mar_18.h5'
with h5py.File(geometry_file, 'r') as f:
    geom = LPDGeometry.from_h5_file_and_quad_positions(f, quadpos)

# run_folder = '/gpfs/exfel/exp/FXE/201701/p002026/proc/'
run_folder = './'
run = RunDirectory(run_folder + runStr)
#run.info()
trainIDs = np.array(run.train_ids,'uint32')
print("Processing data from trainID",trainIDs[takeTrainsBegin], "to train",trainIDs[takeTrainsEnd])

#trainIDs=list()
trainsIDtoFile = list()
cellIDs=list()
pulseIDs=list()
N_set=list()
k=0
#
#tid, train_data = run.train_from_id(trainIDs[takeTrainsBegin])
#print(tid)
#for dev in sorted(train_data.keys()):
#    print(dev, end='\t')
#    try:
#        print(train_data[dev]['image.data'].shape)
#    except KeyError as e:
#        print("No image.data", e)

PulsesPerTrain = 16
k=0

TotalIm = None  # Add by Jun

for trainID in trainIDs[takeTrainsBegin:takeTrainsEnd+1]:
#    print("Processing train",trainID)
    
    tid, train_data = run.train_from_id(trainID)
    NoData = False
    for dev in sorted(train_data.keys()):
#        print(dev, end='\t')
        try:
#            print(train_data[dev]['image.data'].shape)
            train_data[dev]['image.data']
        except KeyError:
            print("No image.data for trainID",trainID)
            NoData = True
            break
    if NoData:
        continue
    
    modules_data = stack_detector_data(train_data, 'image.data')

    # Add by Jun
    # ----------
    if hasattr(modules_data, 'shape') is False or \
            modules_data.shape[-3:] != (16, 256, 256):
        continue

    res, centre = geom.position_all_modules(modules_data)
    # ----------

    if not k:
        TotalIm = np.zeros((res.shape[1],res.shape[2]))   
    assembl_image = np.nan_to_num(res)
    if plotting:
        print(trainID)
        plt.figure(figsize=(11, 11))
        plt.imshow(clip(res[0], max=800))
#        plt.imshow(res[0],vmin=-1, vmax=600)
        plt.show()
#    geom.plot_data(clip(modules_data[0], max=1000))
    mask_data = np.zeros(assembl_image.shape)
    mask_data[assembl_image<=0] = 1
    mask_data[assembl_image>1e4] = 1
    dmask = np.asarray(mask_data, dtype='float16')
#        total_mask = np.asarray(-1*((smask-1)*(dmask-1)-1), dtype='float16')
    total_mask = np.asarray(1*(dmask), dtype='float16')
    TotalIm = TotalIm+np.sum(assembl_image,axis=0)*np.abs(np.array(total_mask[0]-1,dtype='int16'))
    for i in range(0,np.minimum(modules_data.shape[0],PulsesPerTrain)):
        IntStartTime = time.time()
        trainsIDtoFile.append(trainID)
#        TotalIm = TotalIm+assembl_image*np.abs(np.array(total_mask-1,dtype='int16'))

#        if plotting:
#            print(trainID,":",i)
#            plt.figure(figsize=(11, 11))
#            plt.imshow(clip(res[i], max=800))
#    #        plt.imshow(res[0],vmin=-1, vmax=600)
#            plt.show()
        Q,i_unc_sa0 = ai.integrate1d(assembl_image[i],
                          npt,method="BBox",
                          mask=total_mask[i],
                          radial_range=(0.2,5.5),
                          correctSolidAngle=True,
                          polarization_factor=1,
                          unit="q_A^-1")
        if not k:
            q=Q[:,None]
            Sq_sa0=i_unc_sa0[:,None]
        else:
            Sq_sa0=np.concatenate((Sq_sa0,i_unc_sa0[:,None]),axis=1)
        k+=1
        print("Integration took",(time.time()-IntStartTime)*1000,"ms")


if TotalIm is not None:  # Add by Jun
    plt.figure(figsize=(10, 10))
    plt.imshow((TotalIm),vmin=-100, vmax=500000)
    plt.show()
    plt.figure(runNum,figsize=(12,5))
    #    plt.plot(q,np.median(Sq,axis=1),label='median')
    #    plt.plot(q,np.mean(Sq,axis=1),label='mean')
    plt.plot(q,np.median(Sq_sa0,axis=1),label='median')
    plt.plot(q,np.mean(Sq_sa0,axis=1),label='mean')
    plt.legend()
    #    plt.title('Azimuthally integrated Run # ' + str(runNum) + '. Mean and median S(q)')
    #fig.set_size_inches((16,9))
    #    plt.savefig('/gpfs/exfel/exp/FXE/201701/p002026/scratch/ReducedScans/June2018/img_r'+str(runNum).zfill(4)+'.png')
    plt.show()

    TotalTime = (time.time()-startTime)
    print('Whole set took', str(math.ceil(TotalTime)),'s or',str(math.ceil(TotalTime/k*1000)), ' ms per image')