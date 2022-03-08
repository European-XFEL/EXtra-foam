"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .statistics_py import (
    hist_with_stats, nanhist_with_stats, compute_statistics,
    nanmean, nansum, nanstd, nanvar,
    quick_min_max
)

from .miscellaneous import (
    normalize_auc
)
from .sampling import down_sample, slice_curve, up_sample
from .data_structures import (
    OrderedSet, Stack, SimpleSequence, SimpleVectorSequence,
    SimplePairSequence, OneWayAccuPairSequence,
)
from .azimuthal_integ import (
    compute_q, energy2wavelength, AzimuthalIntegrator, ConcentricRingsFinder,
)

from .helpers import intersection

from .imageproc_py import (
    nanmean_image_data, correct_image_data, mask_image_data,
    movingAvgImageData, bin_photons, dropletize
)

from .datamodel import (
    RawImageDataFloat, RawImageDataDouble,
    MovingAverageArrayFloat, MovingAverageArrayDouble,
    MovingAverageFloat, MovingAverageDouble
)

from .spectrum import (
    compute_spectrum_1d
)

from .computer_vision import (
    edge_detect, fourier_transform_2d
)

from .peak_finding import (
    find_peaks_1d
)

from .curve_fitting import (
    CurveFitting, FittingType
)
