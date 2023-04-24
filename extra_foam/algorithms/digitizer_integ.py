#############################################################################
# Author: bermudei
#
# Created on April 18, 2023, 07:14 PM
#
# Copyright (C) European XFEL GmbH Hamburg. All rights reserved.
#############################################################################

import numpy as np
from .peak_finding import find_peaks_1d
from scipy.integrate import trapezoid

# Method to integrate the peak signals of digitizer data.
# Input: Digitizer RAW data (in samples)
#        Width of each peak (in samples) in which to calculate the integral
#        Estimated distance between peaks (in samples)for it to be "counted" as a peak.
# The method first takes the raw digitizer data. Then subtracts the background and makes the signal positive. 
# Afterwards it finds the peaks on the signals (using peak finder) and 
# given the width of the peak, it calculates its integral using the trapezoid method.

def digi_integral(raw_data, pk_width, dist_pk):
    data = np.array(raw_data)
    bckg = np.nanmean(data[0:100])
    digi_data = -data + bckg
    digitizer_peaks = find_peaks_1d(digi_data, height=np.nanmax(digi_data)*0.5, distance=dist_pk)
    idx_digitizer_peaks = digitizer_peaks[0]
    integral_peaks = [trapezoid(digi_data[idx_digitizer_peaks-pk_width:idx_digitizer_peaks+pk_width]) 
                    for idx_digitizer_peaks in idx_digitizer_peaks]
    return integral_peaks
      