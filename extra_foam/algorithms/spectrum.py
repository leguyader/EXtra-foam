"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Loïc Le Guyader <loic.le.guyader@xfel.eu>
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

def compute_spectrum_1d_weighted(x, y, w=None, n_bins=10, *,
                                bin_range=None, edge2center=True,
                                nan_to_num=False):
    """Compute weighted spectrum."""
    if len(x) != len(y):
        raise ValueError(f"x and y have different lengths: "
                         f"{len(x)} and {len(y)}")
    if w is None:
        w = np.ones_like(x)
    elif len(w) != len(x):
        raise ValueError(f"x and w have different lengths: "
                         f"{len(x)} and {len(w)}")
    if len(x) == 0:
        edges = np.full((n_bins + 1,), np.nan)
        counts = np.full((n_bins,), np.nan)
        sum_w = np.full((n_bins,), np.nan)
        sum_w2 = np.full((n_bins,), np.nan)
        wmu = np.full((n_bins,), np.nan)
        t = np.full((n_bins,), np.nan)
        ws = np.full((n_bins,), np.nan)
    else:
        counts, edges, bin_idx = binned_statistic(x, y, 'count',
                                                n_bins, range=bin_range)
        (sum_w, sum_w2, wy), _, _ = binned_statistic(x, (w, w**2, w*y), 'sum',
                                                n_bins, range=bin_range)
        wmu = wy/sum_w
        t, _, _ = binned_statistic(x, w*(y - wmu[bin_idx-1])**2, 'sum',
                                                n_bins, range=bin_range)
        
        ws = np.sqrt(t/(sum_w - sum_w2/sum_w))

    if nan_to_num:
        np.nan_to_num(counts, copy=False)
        np.nan_to_num(sum_w, copy=False)
        np.nan_to_num(sum_w2, copy=False)
        np.nan_to_num(wmu, copy=False)
        np.nan_to_num(t, copy=False)
        np.nan_to_num(ws, copy=False)
        
    if edge2center:
        edges = (edges[1:] + edges[:-1]) / 2.

    stats = {
        'counts': counts,
        'edges': edges,
        'sum_w': sum_w,
        'sum_w2': sum_w2,
        'wmu': wmu,
        't': t,
        'ws': ws}

    return stats

def weighted_incremental_std(xi, wi, sum_w, sum_w2, wmu, t):
    """ Computed the weighted incremental standard deviation with
        reliability weights based on WV2 algorithm from
        West (1979) doi:10.1145/359146.359153
        
        Inputs:
            xi: new value
            wi: new weight
            sum_w: actual sum of weight
            sum_w2: actual sum of square of weight
            wmu: actual weighted mean
            t: actual Sum_i [wi*(xi - wmu)**2]
            
        Outputs:
            sum_w: updated sum of weight
            sum_w2: updated sum of square of weight
            wmu: updated weighted mean
            t: updated Sum_i [wi*(xi - wmu)**2]
            ws: standard deviation with reliability weights
    """
    
    q = xi - wmu
    temp = sum_w + wi
    
    r = q*wi/temp
    wmu += r
    t += r*sum_w*q
    
    sum_w = temp
    sum_w2 = sum_w2 + wi*wi
    if sum_w == sum_w2/sum_w:
        ws = 0
    else:
        ws = np.sqrt(t/(sum_w - sum_w2/sum_w))
    
    return sum_w, sum_w2, wmu, t, ws
