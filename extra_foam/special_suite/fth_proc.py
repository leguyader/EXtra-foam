"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import math

import numpy as np
from scipy import stats

from extra_foam.algorithms import compute_spectrum_1d, nansum
from extra_foam.algorithms import SimpleSequence
from extra_foam.pipeline.processors.binning import _BinMixin
from extra_foam.pipeline.exceptions import ProcessingError
from extra_foam.pipeline.data_model import ImageData

from extra_foam.special_suite.special_analysis_base import profiler, QThreadWorker

from numpy.fft import fft2, fftshift, ifft2, ifftshift
from numpy import abs, isnan, isfinite, zeros_like, ones_like, real

#from weighted_std import compute_spectrum_1d_weighted, weighted_incremental_std

_DEFAULT_N_BINS = 20
_DEFAULT_BIN_RANGE = "-inf, inf"

class FTHProcessor(QThreadWorker, _BinMixin):
    """FTH processor.

    Attributes:
        _image (str): average image
    """

    # 10 pulses/train * 60 seconds * 30 minutes = 18,000
    _MAX_POINTS = 100 * 60 * 60

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._counts1 = None
        self._image = None
        self._data_image = None
        self._fth = None
        self._fth_old = None
        self._mask = None
        self._fth_masked = None
        self._beta = 0.5
        self._window = 20

    def onWindowChanged(self, value: str):
        w = int(value)
        if w != self._window:
            self._window = w

    @profiler("FTH Processor")
    def process(self, data):
        """Override."""
        processed = data["processed"]

        try:
            self._update_data_point(processed, data['raw'])
        except ProcessingError as e:
            self.log.error(repr(e))

        self.log.info(f"Train {processed.tid}: <{self._counts}>")

        return {
            "counts": self._counts,
            "image": self._image,
            "mask": self._mask,
#            "fth_masked": ImageData.from_array(abs(fftshift(self._fth)),
#                image_mask=(fftshift(self._mask) < 1)),
            "fth_masked": ImageData.from_array(abs(fftshift(self._fth)),
                image_mask=(zeros_like(self._fth) < 0)),
            "update_mask_f": self.update_mask
        }

    def _update_data_point(self, processed, raw):
        roi = processed.roi
        masked = processed.image.masked_mean

        # get three sum intensity over kept pulses over ROIs
        image = roi.geom2.rect(masked)
        if image is None:
            raise ProcessingError("ROI2 is not available!")

        if self._image is None:
            self._counts = 1
            self._data_image = image
            self._notnans = isfinite(self._data_image)
            print(f'startup: {self._notnans.shape}')
            self._image = zeros_like(self._data_image)
            self._mask = ones_like(self._data_image)
            self._fth_old = zeros_like(self._data_image)
        else:
            self._counts = self._counts + 1
            n = min(self._counts, self._window)
            self._data_image += (image - self._data_image) / n

        # overwrite data in place
        self._image[self._notnans] = self._data_image[self._notnans]

        # compute auto-correlation
        self._fth = fft2(self._image)

        # use auto-correlation support,
        # inside support, we keep the new guess
        # outside support, we remove from the old guess a beta fraction 
        # of the new guess
        #self._fth[self._mask < 1] = (self._fth_old[self._mask < 1]
        #        - self._beta*self._fth[self._mask < 1])
        # b(n - o) + o = bn + o(1-b)
        #self._fth[self._mask < 1] = (self._beta*self._fth[self._mask < 1]
        #        + (1 - self._beta)*self._fth_old[self._mask < 1])
        self._fth[self._mask < 1] *= 0.75

        # auto-correlation must be positive
        #self._fth[self._fth < 0] = 0

        # save for next iteration
        self._fth_old = self._fth

        # back to hologram space
        back = real(ifft2(self._fth))

        # intensity measured must be positive
        back[back < 0] = 0.0
        
        # fill not measure data
        self._image[~self._notnans] = back[~self._notnans]

    def update_mask(self, x, y, w, h, iserase):
        self.log.info(f'updating mask {x},{y},{w},{h},{iserase}')
        temp = fftshift(self._mask)
        if iserase:
            temp[y:(y+h), x:(x+w)] = 1
        else:
            temp[y:(y+h), x:(x+w),] = 0
        self._mask = ifftshift(temp)

    def reset(self):
        """Override."""
        self._counts = None
        self._data_image = None
        self._image = None
        self._notnans = None
