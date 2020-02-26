import pytest

import math

import numpy as np

from extra_foam.algorithms import (
    compute_roi_hist, compute_statistics, find_actual_range
)


class TestMiscellaneous:

    def testComputeRoiHist(self):
        # case 1
        roi = np.array([[np.nan, 1, 2], [3, 6, np.nan]], dtype=np.float32)
        hist, bin_centers, mean, median, std = compute_roi_hist(roi, (1, 3), 4)
        np.testing.assert_array_equal([1, 0, 1, 1], hist)
        np.testing.assert_array_equal([1.25, 1.75, 2.25, 2.75], bin_centers)
        arr_gt = [1, 2, 3]
        assert np.mean(arr_gt) == mean
        assert np.median(arr_gt) == median
        assert np.std(arr_gt) == pytest.approx(std)

        # case 2 (the actual array is empty after filtering)
        roi = np.array([[np.nan, 0, 0], [0, 0, np.nan]], dtype=np.float32)
        hist, bin_centers, mean, median, std = compute_roi_hist(roi, (1, 3), 4)
        np.testing.assert_array_equal([0, 0, 0, 0], hist)
        assert np.isnan(mean)
        assert np.isnan(median)
        assert np.isnan(std)

        # case 3 (elements in the actual array have the same value)
        roi = np.array([[1, 0, np.nan], [1, np.nan, 0]], dtype=np.float32)
        hist, bin_centers, mean, median, std = compute_roi_hist(roi, (1e-6, 3), 4)
        np.testing.assert_array_equal([0, 2, 0, 0], hist)
        assert 1 == mean
        assert 1 == median
        assert 0 == std

    def testFindActualRange(self):
        arr = np.array([1, 2, 3, 4])
        assert (-1.5, 2.5) == find_actual_range(arr, (-1.5, 2.5))

        arr = np.array([1, 2, 3, 4])
        assert (1, 4) == find_actual_range(arr, (-math.inf, math.inf))

        arr = np.array([1, 1, 1, 1])
        assert (0.5, 1.5) == find_actual_range(arr, (-math.inf, math.inf))

        arr = np.array([1, 2, 3, 4])
        assert (3, 4) == find_actual_range(arr, (3, math.inf))

        arr = np.array([1, 2, 3, 4])
        assert (4, 5) == find_actual_range(arr, (4, math.inf))

        arr = np.array([1, 2, 3, 4])
        assert (5, 6) == find_actual_range(arr, (5, math.inf))

        arr = np.array([1, 2, 3, 4])
        assert (0, 1) == find_actual_range(arr, (-math.inf, 1))

        arr = np.array([1, 2, 3, 4])
        assert (-1, 0) == find_actual_range(arr, (-math.inf, 0))

        arr = np.array([1, 2, 3, 4])
        assert (-1, 0) == find_actual_range(arr, (-math.inf, 0))

    def testComputeStatistics(self):
        with np.warnings.catch_warnings():
            np.warnings.simplefilter("ignore", category=RuntimeWarning)

            # test input contains only Nan
            data = np.empty((3, 2), dtype=np.float32)
            data.fill(np.nan)
            assert all(np.isnan(x) for x in compute_statistics(data))

        data = np.array([])
        assert all(np.isnan(x) for x in compute_statistics(data))

        data = np.array([1, 1, 2, 1, 1])
        assert (1.2, 1.0, 0.4) == compute_statistics(data)
