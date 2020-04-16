"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from scipy.stats import binned_statistic


def compute_spectrum_1d(x, y, n_bins=10, *,
                        bin_range=None, edge2center=True):
    """Compute spectrum."""
    stats, edges, _ = binned_statistic(x, y, 'mean', n_bins, range=bin_range)
    counts, _, _ = binned_statistic(x, y, 'count', n_bins, range=bin_range)
    if edge2center:
        return stats, (edges[1:] + edges[:-1]) / 2., counts
    return stats, edges, counts
