# -*- coding: utf-8 -*-
"""
Created on Sat May 23 21:33:23 2015

@author: khakhulin
"""
import h5py
#import os
#import sys
import numpy as np
#from detect_peaks import detect_peaks
#from scipy import optimize
import scipy
#import pylab

import pyFAI, fabio, time, math
import matplotlib.pyplot as plt
from IPython.display import display, Image
#from numpy import *
import scipy.misc
# from PIL import Image

def h5open(fname):
    try:
        #print("opened %s" % fname)
        h5_file = h5py.File(fname, 'r')
    except:
        h5_file = 0
    return h5_file

def h5close(h5_file):
    h5py.File.close(h5_file)

pi = math.pi;
dist = 160*1e-3 # m
center_X = 580
center_Y = 580
pixel_size = 0.5e-3
energy = 9.33 # keV
wavelength_lambda = 12.3984/energy*1e-10 # m

muSi = 91 # silicon sensor absorption coefficient
tSi = 500*1e-4 # sensor thickness

mus = 3.945 # sample absorption coefficient
ts = 100*1e-4 # sample thickness

HoleSize = -26.28e-3
HoleSize_pixels = np.abs(np.int(np.ceil(HoleSize/pixel_size)))

Q_offset = 3;

SM_unit = 256
dx_map = [0, 0, SM_unit, SM_unit, 0, 0, SM_unit, SM_unit, SM_unit*2, SM_unit*2,
          SM_unit*3, SM_unit*3, SM_unit*2, SM_unit*2, SM_unit*3, SM_unit*3,]
dy_map = [0, SM_unit, SM_unit, 0, SM_unit*2, SM_unit*3, SM_unit*3, SM_unit*2,
          SM_unit*2, SM_unit*3, SM_unit*3, SM_unit*2, 0, SM_unit, SM_unit, 0]

FullIm = np.zeros([SM_unit*4,SM_unit*4],dtype='uint16')
CombFullIm = np.zeros([SM_unit*4+HoleSize_pixels+Q_offset,SM_unit*4+HoleSize_pixels+Q_offset],dtype='uint16')

# setting the integrator
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
Qnorm_max = 3

DarkGain100 = np.zeros([256,256],dtype='uint16')
DarkGain10 = 4096*np.ones([256,256],dtype='uint16')
DarkGain1 = 8192*np.ones([256,256],dtype='uint16')

# LPD run number
runNum = 79
#TODO Q: DO you prefer upper bounds inclusive or exclusive? Currently inclusive.
setdID_lower = 0
sedID_upper = setdID_lower
pulseIDs = range(0,3)
alwaysIndexOverSets = False
inDirName ='/gpfs/exfel/data/group/cas/waxint/r{0:04}/'\

indexOverSets = alwaysIndexOverSets or sedID_upper != setdID_lower
setIDs = range(setdID_lower, sedID_upper + 1)
firstTime = True

if setdID_lower > sedID_upper:
    raise Exception("Lower limit of set ID must be less than or equal to upper limit.")
elif setdID_lower == sedID_upper:
    foutname='INT-R{:04}-S{:05}.h5'.format(runNum, setdID_lower)
else:
    foutname='INT-R{:04}-S{:05}_S{:05}.h5'.format(runNum, setdID_lower, sedID_upper)

print(foutname)

with h5py.File(foutname, 'w') as outFile:
#outFile = h5py.File(foutname, 'w')
  # loop over blocks (data set files)
  for setID in setIDs:
    #loop over pulses (or memory cells)
    for pulseID in pulseIDs:
        #loop over the supermodules to assemble images
        read_startTime = time.time()
        for i in range(0,16):
            #fname='/gpfs/p900002/raw/r{0:04}/'\
            fname=inDirName + \
            'RAW-R{0:04}-LPD{1:02}-S{2:05}.h5'.format(runNum, i, setID)
            fname_dark100='/home/khakhuli/AzimuthalIntegration/'+\
            'DarkGain100_LPD{:02}.h5'.format(i)
            fname_dark10='/home/khakhuli/AzimuthalIntegration/'+\
            'DarkGain10_LPD{:02}.h5'.format(i)
            fname_dark1='/home/khakhuli/AzimuthalIntegration/'+\
            'DarkGain1_LPD{:02}.h5'.format(i)

            h5_file = h5open(fname)
            if h5_file!=0:
                print("Opened file {}".format(fname))
                h5path_n2Dimages = '/INSTRUMENT/FXE_DET_LPD1M-1/DET/'+str(i)+ \
                'CH0:xtdf/image/data'
                image_stack = h5_file[h5path_n2Dimages]
                CurrIm = np.array(image_stack[pulseID][0][:][:],dtype='uint16')
                CurrIm_np = np.array(CurrIm,dtype='uint16')

                # Don't have access to these so default values used
                try:
                    h5dark_file = h5open(fname_dark1)
                    DarkGain1 = np.array(h5dark_file['DarkImages'])[pulseID][:][:]
                    h5close(h5dark_file)

                    h5dark_file = h5open(fname_dark10)
                    DarkGain10 = np.array(h5dark_file['DarkImages'])[pulseID][:][:]
                    h5close(h5dark_file)

                    h5dark_file = h5open(fname_dark100)
                    DarkGain100 = np.array(h5dark_file['DarkImages'])[pulseID][:][:]
                    h5close(h5dark_file)

                    CorrIm = CurrIm_np - DarkGain100
                    CurrIm[np.where(CurrIm_np<=4096)] = CorrIm[np.where(CurrIm_np<=4096)]

                    CorrIm = (CurrIm_np - DarkGain10)*10

                    CurrIm[np.where(np.logical_and(CurrIm_np<=8192,CurrIm_np>4096))] \
                    = CorrIm[np.where(np.logical_and(CurrIm_np<=8192,CurrIm_np>4096))]

                    CorrIm = (CurrIm_np - DarkGain1)*100
                    CurrIm[np.where(CurrIm_np>8192)] = CorrIm[np.where(CurrIm_np>8192)]
                except:
                    print("Failed to find dark gain files.")

            else:
                print("Couldn't open {}".format(fname))
                CurrIm = np.zeros([SM_unit,SM_unit],dtype='uint16')
            FullIm[dy_map[i]:dy_map[i]+SM_unit,SM_unit*4-dx_map[i]-SM_unit:SM_unit*4-dx_map[i]] \
            =    np.rot90(CurrIm,2)

            if HoleSize>0:
                CombFullIm[0:SM_unit*2,HoleSize_pixels:HoleSize_pixels+SM_unit*2]=\
                FullIm[0:SM_unit*2,0:SM_unit*2]
                #
                CombFullIm[SM_unit*2+HoleSize_pixels:SM_unit*4+HoleSize_pixels,SM_unit*2:SM_unit*4]=\
                FullIm[SM_unit*2:SM_unit*4,SM_unit*2:SM_unit*4]
                #
                CombFullIm[HoleSize_pixels:SM_unit*2+HoleSize_pixels,HoleSize_pixels+SM_unit*2:HoleSize_pixels+SM_unit*4]=\
                FullIm[0:SM_unit*2,SM_unit*2:SM_unit*4]

                CombFullIm[SM_unit*2:SM_unit*4,0:SM_unit*2]=\
                FullIm[SM_unit*2:SM_unit*4,0:SM_unit*2]
            else:
                #Q1
                CombFullIm[0:SM_unit*2,SM_unit*2+Q_offset:SM_unit*4+Q_offset]=\
                FullIm[0:SM_unit*2,SM_unit*2:SM_unit*4]
                #Q2:
                CombFullIm[SM_unit*2+Q_offset:SM_unit*4+Q_offset,SM_unit*2+HoleSize_pixels+Q_offset:SM_unit*4+HoleSize_pixels+Q_offset]=\
                FullIm[SM_unit*2:SM_unit*4,SM_unit*2:SM_unit*4]
                #Q3
                CombFullIm[SM_unit*2+HoleSize_pixels+Q_offset:SM_unit*4+HoleSize_pixels+Q_offset,HoleSize_pixels:SM_unit*2+HoleSize_pixels]=\
                FullIm[SM_unit*2:SM_unit*4,0:SM_unit*2]
                #Q4:
                CombFullIm[HoleSize_pixels:HoleSize_pixels+SM_unit*2,0:SM_unit*2]=\
                FullIm[0:SM_unit*2,0:SM_unit*2]


        print('Full Image Set #'+str(setID+1)+ '; Pulse #'+str(pulseID+1)+ \
              ' prepared in '+ str(math.ceil((time.time()-read_startTime)*1e3)) +' ms')
        """plt.figure(figsize=(10,10))
        #plt.imshow(FullIm)
        plt.imshow(FullIm,vmin=-10, vmax=8000)
        plt.colorbar()
        plt.show()

        #Final assembled image is ready, we display and start pyFAI integartion
        plt.figure(figsize=(10,10))
        plt.imshow(CombFullIm,vmin=-10, vmax=7000)
        plt.colorbar()
        plt.show()"""

        int_startTime = time.time()

        cm_correct = CombFullIm
        mask_data = np.zeros(cm_correct.shape)

        #cm_correct = CurrIm;
        Q,i_unc = ai.integrate1d(cm_correct,
                                  npt,method="lut",
                                  radial_range = [0,3],
                                  mask=mask_data,
                                  correctSolidAngle=True,
                                  polarization_factor=1,
                                  unit="q_A^-1")

        I_unc = i_unc[:,None]
        integrationTime = ((time.time()-int_startTime)*1000)
        print('Azimuthal integration of image took', str(math.ceil(integrationTime)), ' ms')

        q=Q[:,None]
        tth = np.rad2deg(2*np.arcsin(q*wavelength_lambda*1e10/(4*pi))) # 2-theta scattering angle
        T_Si = (1-np.exp(-muSi*tSi))/(1-np.exp(-muSi*tSi/np.cos(np.deg2rad(tth)))) # silicon sensor absorption correction
        Ts = 1/(mus*ts)*np.cos(np.deg2rad(tth))/(1-np.cos(np.deg2rad(tth)))*(np.exp(-mus*ts)-np.exp(-mus*ts/np.cos(np.deg2rad(tth))))
        Ts = Ts/Ts[0] # sample absorption correction in isotropic case (not to do here if done on the image before integration)
        Qnorm=Q[np.where(np.logical_and(Q>=Qnorm_min,Q<=Qnorm_max))]
        # normalizing:

        N = np.trapz(i_unc[np.where(np.logical_and(Q>=Qnorm_min,Q<=Qnorm_max))],x=Qnorm)
        # appying the corrections
        I_cor = I_unc*T_Si#/Ts

        # correct the shape - make 1-d
        q.shape=(q.shape[0],)
        I_cor.shape=(I_cor.shape[0],)

        """plt.figure(20)
        plt.plot(q,I_cor)
        plt.show()"""

        dataPath = 'DATA'
        if indexOverSets:
            if firstTime:
                firstTime = False
                dims = (len(setIDs), len(pulseIDs), len(q))
                ds = outFile.create_dataset(dataPath + '/Q', dims)
                ds_cor = outFile.create_dataset(dataPath + '/I_CORRECTED', dims)
                ds_unc = outFile.create_dataset(dataPath + '/I_UNCORRECTED', dims)
            ds[setID, pulseID] = q
            ds_cor[setID, pulseID] = I_cor
            ds_unc[setID, pulseID] = i_unc
        else:
            if firstTime:
                firstTime = False
                dims = (len(pulseIDs), len(q))
                ds = outFile.create_dataset(dataPath + '/Q', dims)
                ds_cor = outFile.create_dataset(dataPath + '/I_CORRECTED', dims)
                ds_unc = outFile.create_dataset(dataPath + '/I_UNCORRECTED', dims)
            ds[pulseID] = q
            ds_cor[pulseID] = I_cor
            ds_unc[pulseID] = i_unc

  statsPath = 'DATA/STATS/I_CORRECTED'
  dataPath = 'DATA/I_CORRECTED'
  data = outFile[dataPath]
  if indexOverSets:
    pass
    # TODO do we aggregate over just pulse or pulses and sets?
    outFile[statsPath + '/MEAN'] = np.mean(data.value, axis=(0,1))
    outFile[statsPath + '/MEDIAN'] = np.median(data, axis=(0,1))
    outFile[statsPath + '/STD'] = np.std(data, axis=(0,1))
  else:
    outFile[statsPath + '/MEAN'] = np.mean(data.value, axis=0)
    outFile[statsPath + '/MEDIAN'] = np.median(data, axis=0)
    outFile[statsPath + '/STD'] = np.std(data, axis=0)
