"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Loïc Le Guyader <loic.le.guyader@xfel.eu>
Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import math

import numpy as np
from numpy import isfinite
from scipy import stats

from extra_foam.algorithms import (compute_spectrum_1d, nansum,
    compute_spectrum_1d_weighted, weighted_incremental_std)

from extra_foam.algorithms import SimpleSequence
from extra_foam.pipeline.processors.binning import _BinMixin
from extra_foam.pipeline.exceptions import ProcessingError

from extra_foam.special_suite.special_analysis_base import profiler, QThreadWorker

_DEFAULT_N_BINS = 20
_DEFAULT_BIN_RANGE = "-inf, inf"

class TrXasProcessor(QThreadWorker, _BinMixin):
    """Time-resolved XAS processor.

    The implementation of tr-XAS processor is easier than bin processor
    since it cannot have empty device ID or property. Moreover, it does
    not include VFOM heatmap.

    Absorption ROI-i/ROI-j is defined as -log(sum(ROI-i)/sum(ROI-j)).

    Attributes:
        _device_id1 (str): device ID 1.
        _ppt1 (str): property of device 1.
        _device_id2 (str): device ID 2.
        _ppt2 (str): property of device 2.
        _slow1 (SimpleSequence): store train-resolved data of source 1.
        _slow2 (SimpleSequence): store train-resolved data of source 2.
        _t13 (SimpleSequence): store train-resolved transmission ROI1/ROI3.
        _t23 (SimpleSequence): store train-resolved transmission ROI2/ROI3.
        _t21 (SimpleSequence): store train-resolved transmission ROI2/ROI1.
        _r1 (SimpleSequence): store train-resolved ROI1 intensities.
        _r2 (SimpleSequence): store train-resolved ROI2 intensities.
        _r3 (SimpleSequence): store train-resolved ROI3 intensities.
        _tid (SimpleSequence): store train IDs.
        _edges1 (numpy.array): edges of bin 1. shape = (_n_bins1 + 1,)
        _counts1 (numpy.array): counts of bin 1. shape = (_n_bins1,)
        _t13_stats (numpy.array): dict of stats of transmission ROI1/ROI3 with
            respect to source 1.
        _t23_stats (numpy.array): dict of stats of transmission ROI2/ROI3 with
            respect to source 1.
        _t21_stats (numpy.array): dict of stats of transmission ROI2/ROI1 with
            respect to source 1.
        _edges2 (numpy.array): edges of bin 2. shape = (_n_bins2 + 1,)
        _a21_heat (numpy.array): 2D binning of absorption ROI2/ROI1.
            shape = (_n_bins2, _n_bins1)
        _a21_heat_count (numpy.array): counts of 2D binning of absorption
            ROI2/ROI1. shape = (_n_bins2, _n_bins1)
        _bin_range1 (tuple): bin 1 range requested.
        _actual_range1 (tuple): actual bin range used in bin 1.
        _n_bins1 (int): number of bins of bin 1.
        _bin_range2 (tuple): bin 2 range requested.
        _actual_range2 (tuple): actual bin range used in bin 2.
        _n_bins2 (int): number of bins of bin 2.
        _bin1d (bool): a flag indicates whether data need to be re-binned
            with respect to source 1.
        _bin2d (bool): a flag indicates whether data need to be re-binned
            with respect to both source 1 and source 2.
        _reset (bool): True for clearing all the existing data.
    """

    # 10 pulses/train * 60 seconds * 30 minutes = 18,000
    _MAX_POINTS = 100 * 60 * 60

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._device_id1 = ""
        self._ppt1 = ""
        self._device_id2 = ""
        self._ppt2 = ""

        self._slow1 = SimpleSequence(max_len=self._MAX_POINTS)
        self._slow2 = SimpleSequence(max_len=self._MAX_POINTS)
        self._t13 = SimpleSequence(max_len=self._MAX_POINTS)
        self._t23 = SimpleSequence(max_len=self._MAX_POINTS)
        self._t21 = SimpleSequence(max_len=self._MAX_POINTS)

        self._r1 = SimpleSequence(max_len=self._MAX_POINTS)
        self._r2 = SimpleSequence(max_len=self._MAX_POINTS)
        self._r3 = SimpleSequence(max_len=self._MAX_POINTS)
        self._tid = SimpleSequence(max_len=self._MAX_POINTS)
        self._saturation = SimpleSequence(max_len=self._MAX_POINTS)

        self._edges1 = None
        self._counts1 = None
        self._t13_stats = None
        self._t23_stats = None
        self._t21_stats = None

        self._time_edges1 = None
        self._time_counts1 = None
        self._time_t13_stats = None
        self._time_t23_stats = None
        self._time_t21_stats = None
        self._time_saturation_stats = None

        self._edges2 = None
        self._a21_heat = None
        self._a21_heat_count = None

        self._bin_range1 = self.str2range(_DEFAULT_BIN_RANGE)
        self._actual_range1 = None
        self._auto_range1 = [True, True]
        self._n_bins1 = _DEFAULT_N_BINS
        self._bin_range2 = self.str2range(_DEFAULT_BIN_RANGE)
        self._actual_range2 = None
        self._auto_range2 = [True, True]
        self._n_bins2 = _DEFAULT_N_BINS

        self._time_bin_range1 = self.str2range(_DEFAULT_BIN_RANGE)
        self._time_actual_range1 = None
        self._time_auto_range1 = [True, True]
        self._time_bin_width1 = _DEFAULT_N_BINS
        self._time_n_bins1 = 1

        self._bin1d = True
        self._bin2d = True

        self._time_bin1d = True

    def onDeviceId1Changed(self, value: str):
        self._device_id1 = value

    def onProperty1Changed(self, value: str):
        self._ppt1 = value

    def onDeviceId2Changed(self, value: str):
        self._device_id2 = value

    def onProperty2Changed(self, value: str):
        self._ppt2 = value

    def onNBins1Changed(self, value: str):
        n_bins = int(value)
        if n_bins != self._n_bins1:
            self._n_bins1 = n_bins
            self._bin1d = True
            self._bin2d = True

    def onBinRange1Changed(self, value: tuple):
        if value != self._bin_range1:
            self._bin_range1 = value
            self._auto_range1[:] = [math.isinf(v) for v in value]
            self._bin1d = True
            self._bin2d = True

    def onNBins2Changed(self, value: str):
        n_bins = int(value)
        if n_bins != self._n_bins2:
            self._n_bins2 = n_bins
            self._bin2d = True

    def onBinRange2Changed(self, value: tuple):
        if value != self._bin_range2:
            self._bin_range2 = value
            self._auto_range2[:] = [math.isinf(v) for v in value]

    def onNTimeBinWidthChanged(self, value: str):
        w = int(value)
        if w != self._time_bin_width1:
            self._time_bin_width1 = w
            self._time_bin1d = True

    def sources(self):
        """Override."""
        return [
            (self._device_id1, self._ppt1, 0),
            (self._device_id2, self._ppt2, 0),
        ]

    @profiler("tr-XAS Processor")
    def process(self, data):
        """Override."""
        processed = data["processed"]

        roi1, roi2, roi3 = None, None, None
        t13, t23, t21 = None, None, None
        r1, r2, r3, s1, s2 = None, None, None, None, None
        sat, tid = None, None
        try:
            roi1, roi2, roi3, t13, t23, t21, r1, r2, r3, s1, s2, sat, tid = \
                self._update_data_point(processed, data['raw'])
        except ProcessingError as e:
            self.log.error(repr(e))

        actual_range1 = self.get_actual_range(
            self._slow1.data(), self._bin_range1, self._auto_range1)
        if actual_range1 != self._actual_range1:
            self._actual_range1 = actual_range1
            self._bin1d = True
            self._bin2d = True

        if self._bin1d:
            self._new_1d_binning()
            self._bin1d = False
        else:
            if t21 is not None:
                self._update_1d_binning(t13, t23, t21, r1, r2, r3, s1, sat, tid)

        actual_range2 = self.get_actual_range(
            self._slow2.data(), self._bin_range2, self._auto_range2)
        if actual_range2 != self._actual_range2:
            self._actual_range2 = actual_range2
            self._bin2d = True

        if self._bin2d:
            self._new_2d_binning()
            self._bin2d = False
        else:
            if t21 is not None:
                self._update_2d_binning(t21, s1, s2)

        time_actual_range1 = self.get_actual_range(
            self._tid.data(), self._time_bin_range1, self._time_auto_range1)
        if time_actual_range1 != self._time_actual_range1:
            self._time_actual_range1 = time_actual_range1
            self._time_n_bins1 = (time_actual_range1[1] - time_actual_range1[0])//self._time_bin_width1 + 1
            self._time_bin1d = True

        if self._time_bin1d:
            self._new_1d_time_binning()
            self._time_bin1d = False
        else:
            self._update_1d_time_binning(t13, t23, t21, r1, r2, r3, s1, sat, tid)

        self.log.info(f"Train {processed.tid} processed")

        # clean up nan and inf
        log13 = -np.log(self._t13_stats['wmu'])
        log13[~isfinite(log13)] = 0
        log23 = -np.log(self._t23_stats['wmu'])
        log23[~isfinite(log23)] = 0
        log21 = -np.log(self._t21_stats['wmu'])
        log21[~isfinite(log21)] = 0

        snr13 = self._t13_stats['wmu']/self._t13_stats['ws']
        snr13[~isfinite(snr13)] = 0
        snr23 = self._t23_stats['wmu']/self._t23_stats['ws']
        snr23[~isfinite(snr23)] = 0
        snr21 = self._t21_stats['wmu']/self._t21_stats['ws']
        snr21[~isfinite(snr21)] = 0

        ret = {
            "rois": np.hstack((roi1, roi2, roi3)),
            #"roi1": roi1,
            #"roi2": roi2,
            #"roi3": roi3,
            "centers1": self.edges2centers(self._edges1)[0],
            "counts1": self._counts1,
            "centers2": self.edges2centers(self._edges2)[0],
            "log_wmu13": log13,
            "log_wmu23": log23,
            "log_wmu21": log21,
            "snr13": snr13,
            "snr23": snr23,
            "snr21": snr21,
            "a21_heat": self._a21_heat,
            "a21_heat_count": self._a21_heat_count,

            "time_snr13": self._time_t13_stats['wmu']/self._time_t13_stats['ws'],
            "time_snr23": self._time_t23_stats['wmu']/self._time_t23_stats['ws'],
            "time_snr21": self._time_t21_stats['wmu']/self._time_t21_stats['ws'],
            "time_saturation": self._time_saturation_stats,
            "time_centers1": self.edges2centers(self._time_edges1)[0]
        }
        
        return ret


    def _update_data_point(self, processed, raw):
        roi = processed.roi
        masked = processed.image.masked_mean

        N_pulses = processed.image.n_images
        kept_pulses = processed.pidx.kept_indices(N_pulses)
        N_kept_pulses = len(kept_pulses)
        sat = 1.0 - N_kept_pulses/N_pulses
        tid = processed.tid

        # get three sum intensity over kept pulses over ROIs
        roi1 = N_kept_pulses*roi.geom1.rect(masked)
        if roi1 is None:
            raise ProcessingError("ROI1 is not available!")
        roi2 = N_kept_pulses*roi.geom2.rect(masked)
        if roi2 is None:
            raise ProcessingError("ROI2 is not available!")
        roi3 = N_kept_pulses*roi.geom3.rect(masked)
        if roi3 is None:
            raise ProcessingError("ROI3 is not available!")

        # get sums of the three ROIs
        r1 = nansum(roi1)
        if r1 <= 0:
            raise ProcessingError("ROI1 sum <= 0!")
        r2 = nansum(roi2)
        if r2 <= 0:
            raise ProcessingError("ROI2 sum <= 0!")
        r3 = nansum(roi3)
        if r3 <= 0:
            raise ProcessingError("ROI3 sum <= 0!")

        # calculate transmissions
        t13 = r1/r3
        t23 = r2/r3
        t21 = r2/r1

        # update historic data
        self._r1.append(r1)
        self._r2.append(r2)
        self._r3.append(r3)
        self._t13.append(t13)
        self._t23.append(t23)
        self._t21.append(t21)
        self._tid.append(tid)
        self._saturation.append(sat)

        # fetch slow data
        s1 = self.getPropertyData(raw, self._device_id1, self._ppt1)
        self._slow1.append(s1)
        s2 = self.getPropertyData(raw, self._device_id2, self._ppt2)
        self._slow2.append(s2)

        return roi1, roi2, roi3, t13, t23, t21, r1, r2, r3, s1, s2, sat, tid

    def _new_1d_binning(self):
        self._t13_stats = compute_spectrum_1d_weighted(
            self._slow1.data(),
            self._t13.data(),
            self._r3.data(),
            n_bins=self._n_bins1,
            bin_range=self._actual_range1,
            edge2center=False,
            nan_to_num=True
        )

        self._t23_stats = compute_spectrum_1d_weighted(
            self._slow1.data(),
            self._t23.data(),
            self._r3.data(),
            n_bins=self._n_bins1,
            bin_range=self._actual_range1,
            edge2center=False,
            nan_to_num=True
        )

        self._t21_stats = compute_spectrum_1d_weighted(
            self._slow1.data(),
            self._t21.data(),
            self._r1.data(),
            n_bins=self._n_bins1,
            bin_range=self._actual_range1,
            edge2center=False,
            nan_to_num=True
        )
        self._edges1 = self._t21_stats['edges']
        self._counts1 = self._t21_stats['counts']
 
    def _new_1d_time_binning(self):
        self._time_saturation_stats, _, _ = compute_spectrum_1d(
            self._tid.data(),
            self._saturation.data(),
            n_bins=self._time_n_bins1,
            bin_range=self._time_actual_range1,
            edge2center=False,
            nan_to_num=True
        )

        self._time_t13_stats = compute_spectrum_1d_weighted(
            self._tid.data(),
            self._t13.data(),
            self._r3.data(),
            n_bins=self._time_n_bins1,
            bin_range=self._time_actual_range1,
            edge2center=False,
            nan_to_num=True
        )

        self._time_t23_stats = compute_spectrum_1d_weighted(
            self._tid.data(),
            self._t23.data(),
            self._r3.data(),
            n_bins=self._time_n_bins1,
            bin_range=self._time_actual_range1,
            edge2center=False,
            nan_to_num=True
        )

        self._time_t21_stats = compute_spectrum_1d_weighted(
            self._tid.data(),
            self._t21.data(),
            self._r1.data(),
            n_bins=self._time_n_bins1,
            bin_range=self._time_actual_range1,
            edge2center=False,
            nan_to_num=True
        )
        self._time_edges1 = self._time_t21_stats['edges']
        self._time_counts1 = self._time_t21_stats['counts']
 
    def _update_1d_binning(self, t13, t23, t21, r1, r2, r3, delay, sat, tid):
        iloc_x = self.searchsorted(self._edges1, delay)
        if 0 <= iloc_x < self._n_bins1:
            self._counts1[iloc_x] += 1
            """
            count = self._counts1[iloc_x]
            self._a13_stats[iloc_x] += (a13 - self._a13_stats[iloc_x]) / count
            self._a23_stats[iloc_x] += (a23 - self._a23_stats[iloc_x]) / count
            self._a21_stats[iloc_x] += (a21 - self._a21_stats[iloc_x]) / count
            """

            sum_w, sum_w2, wmu, t, ws = weighted_incremental_std(
                t13, r3, self._t13_stats['sum_w'][iloc_x],
                self._t13_stats['sum_w2'][iloc_x], 
                self._t13_stats['wmu'][iloc_x],
                self._t13_stats['t'][iloc_x])
            self._t13_stats['sum_w'][iloc_x] = sum_w
            self._t13_stats['sum_w'][iloc_x] = sum_w2
            self._t13_stats['wmu'][iloc_x] = wmu
            self._t13_stats['t'][iloc_x] = t
            self._t13_stats['ws'][iloc_x] = ws
            self._t13_stats['counts'][iloc_x] += 1

            sum_w, sum_w2, wmu, t, ws = weighted_incremental_std(
                t23, r3, self._t23_stats['sum_w'][iloc_x],
                self._t23_stats['sum_w2'][iloc_x], 
                self._t23_stats['wmu'][iloc_x],
                self._t23_stats['t'][iloc_x])
            self._t23_stats['sum_w'][iloc_x] = sum_w
            self._t23_stats['sum_w'][iloc_x] = sum_w2
            self._t23_stats['wmu'][iloc_x] = wmu
            self._t23_stats['t'][iloc_x] = t
            self._t23_stats['ws'][iloc_x] = ws
            self._t23_stats['counts'][iloc_x] += 1

            sum_w, sum_w2, wmu, t, ws = weighted_incremental_std(
                t21, r1, self._t21_stats['sum_w'][iloc_x],
                self._t21_stats['sum_w2'][iloc_x], 
                self._t21_stats['wmu'][iloc_x],
                self._t21_stats['t'][iloc_x])
            self._t21_stats['sum_w'][iloc_x] = sum_w
            self._t21_stats['sum_w'][iloc_x] = sum_w2
            self._t21_stats['wmu'][iloc_x] = wmu
            self._t21_stats['t'][iloc_x] = t
            self._t21_stats['ws'][iloc_x] = ws
            self._t21_stats['counts'][iloc_x] += 1

    def _update_1d_time_binning(self, t13, t23, t21, r1, r2, r3, delay, sat, tid):
        iloc_x = self.searchsorted(self._time_edges1, tid)
        if 0 <= iloc_x < self._time_n_bins1:
            self._time_counts1[iloc_x] += 1
            count = self._time_counts1[iloc_x]
            self._time_saturation_stats[iloc_x] += (sat
                - self._time_saturation_stats[iloc_x]) / count

            sum_w, sum_w2, wmu, t, ws = weighted_incremental_std(
                t13, r3, self._time_t13_stats['sum_w'][iloc_x],
                self._time_t13_stats['sum_w2'][iloc_x], 
                self._time_t13_stats['wmu'][iloc_x],
                self._time_t13_stats['t'][iloc_x])
            self._time_t13_stats['sum_w'][iloc_x] = sum_w
            self._time_t13_stats['sum_w'][iloc_x] = sum_w2
            self._time_t13_stats['wmu'][iloc_x] = wmu
            self._time_t13_stats['t'][iloc_x] = t
            self._time_t13_stats['ws'][iloc_x] = ws
            self._time_t13_stats['counts'][iloc_x] += 1

            sum_w, sum_w2, wmu, t, ws = weighted_incremental_std(
                t23, r3, self._time_t23_stats['sum_w'][iloc_x],
                self._time_t23_stats['sum_w2'][iloc_x], 
                self._time_t23_stats['wmu'][iloc_x],
                self._time_t23_stats['t'][iloc_x])
            self._time_t23_stats['sum_w'][iloc_x] = sum_w
            self._time_t23_stats['sum_w'][iloc_x] = sum_w2
            self._time_t23_stats['wmu'][iloc_x] = wmu
            self._time_t23_stats['t'][iloc_x] = t
            self._time_t23_stats['ws'][iloc_x] = ws
            self._time_t23_stats['counts'][iloc_x] += 1

            sum_w, sum_w2, wmu, t, ws = weighted_incremental_std(
                t21, r1, self._time_t21_stats['sum_w'][iloc_x],
                self._time_t21_stats['sum_w2'][iloc_x], 
                self._time_t21_stats['wmu'][iloc_x],
                self._time_t21_stats['t'][iloc_x])
            self._time_t21_stats['sum_w'][iloc_x] = sum_w
            self._time_t21_stats['sum_w'][iloc_x] = sum_w2
            self._time_t21_stats['wmu'][iloc_x] = wmu
            self._time_t21_stats['t'][iloc_x] = t
            self._time_t21_stats['ws'][iloc_x] = ws
            self._time_t21_stats['counts'][iloc_x] += 1

    def _new_2d_binning(self):
        # to have energy on x axis and delay on y axis
        # Note: the return array from 'stats.binned_statistic_2d' has a swap x and y
        # axis compared to conventional image data
        self._a21_heat, _, self._edges2, _ = \
            stats.binned_statistic_2d(self._slow1.data(),
                                      self._slow2.data(),
                                      -np.log(self._t21.data()),
                                      'mean',
                                      [self._n_bins1, self._n_bins2],
                                      [self._actual_range1, self._actual_range2])
        np.nan_to_num(self._a21_heat, copy=False)

        self._a21_heat_count, _, _, _ = \
            stats.binned_statistic_2d(self._slow1.data(),
                                      self._slow2.data(),
                                      -np.log(self._t21.data()),
                                      'count',
                                      [self._n_bins1, self._n_bins2],
                                      [self._actual_range1, self._actual_range2])
        np.nan_to_num(self._a21_heat_count, copy=False)

    def _update_2d_binning(self, t21, energy, delay):
        iloc_x = self.searchsorted(self._edges2, energy)
        iloc_y = self.searchsorted(self._edges1, delay)
        if 0 <= iloc_x < self._n_bins2 \
                and 0 <= iloc_y < self._n_bins1:
            self._a21_heat_count[iloc_y, iloc_x] += 1
            self._a21_heat[iloc_y, iloc_x] += \
                (-np.log(t21) - self._a21_heat[iloc_y, iloc_x]) / \
                self._a21_heat_count[iloc_y, iloc_x]

    def reset(self):
        """Override."""
        self._slow1.reset()
        self._slow2.reset()
        self._t13.reset()
        self._t23.reset()
        self._t21.reset()
        self._r1.reset()
        self._r2.reset()
        self._r3.reset()
        self._tid.reset()

        self._edges1 = None
        self._counts1 = None
        self._t13_stats = None
        self._t23_stats = None
        self._t21_stats = None
        self._edges2 = None
        self._a21_heat = None
        self._a21_heat_count = None

        self._time_saturation_stats = None
        self._time_t13_stats = None
        self._time_t23_stats = None
        self._time_t21_stats = None
        self._time_edges1 = None
        self._time_counts1 = None

        self._bin1d = True
        self._bin2d = True
        self._time_bin1d = True
