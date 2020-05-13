"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np
from scipy.stats import binned_statistic


def compute_spectrum_1d(x, y, n_bins=10, *,
                        bin_range=None, edge2center=True, nan_to_num=False):
    """Compute spectrum."""
    if len(x) != len(y):
        raise ValueError(f"x and y have different lengths: "
                         f"{len(x)} and {len(y)}")

    if len(x) == 0:
        stats = np.full((n_bins,), np.nan)
        edges = np.full((n_bins + 1,), np.nan)
        counts = np.full((n_bins,), np.nan)
    else:
        stats, edges, _ = binned_statistic(x, y, 'mean', n_bins, range=bin_range)
        counts, _, _ = binned_statistic(x, y, 'count', n_bins, range=bin_range)

    if nan_to_num:
        np.nan_to_num(stats, copy=False)
        np.nan_to_num(counts, copy=False)

    if edge2center:
        return stats, (edges[1:] + edges[:-1]) / 2., counts
    return stats, edges, counts
