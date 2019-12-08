"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from concurrent.futures import ThreadPoolExecutor
import functools

import numpy as np
from scipy import constants

from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

from .base_processor import _BaseProcessor
from ..data_model import MovingAverageArray
from ...algorithms import mask_image, slice_curve
from ...config import Normalizer, AnalysisType
from ...database import Metadata as mt
from ...utils import profiler


def energy2wavelength(energy):
    # Plank-einstein relation (E=hv)
    HC_E = 1e-3 * constants.c * constants.h / constants.e
    return HC_E / energy


class _AzimuthalIntegrationProcessorBase(_BaseProcessor):
    """Base class for AzimuthalIntegrationProcessors.

    Attributes:
        _sample_dist (float): distance from the sample to the
            detector plan (orthogonal distance, not along the beam),
            in meter.
        _pixel1 (float): pixel size along axis 1 in meter.
        _pixel2 (float): pixel size along axis 2 in meter.
        _poni1 (float): poni1 in meter.
        _poni2 (float): poni2 in meter.
        _wavelength (float): photon wavelength in meter.
        _integ_method (string): the azimuthal integration
            method supported by pyFAI.
        _integ_range (tuple): the lower and upper range of
            the integration radial unit. (float, float)
        _integ_points (int): number of points in the
            integration output pattern.
        _normalizer (int): normalizer type for calculating FOM from
            azimuthal integration result.
        _auc_range (tuple): x range for calculating AUC, which is used as
            a normalizer of the azimuthal integration.
        _fom_integ_range (tuple): integration range for calculating FOM from
            the normalized azimuthal integration.
        _integrator (AzimuthalIntegrator): AzimuthalIntegrator instance.
        _ma_window (int): moving average window size.
    """

    def __init__(self):
        super().__init__()

        self.analysis_type = AnalysisType.UNDEFINED

        self._sample_dist = None
        self._pixel1 = None
        self._pixel2 = None
        self._poni1 = None
        self._poni2 = None
        self._wavelength = None

        self._integ_method = None
        self._integ_range = None
        self._integ_points = None

        self._normalizer = None
        self._auc_range = None
        self._fom_integ_range = None

        self._integrator = None

        self._reset_ma = False

    def update(self):
        """Override."""
        g_cfg = self._meta.hget_all(mt.GLOBAL_PROC)
        self._sample_dist = float(g_cfg['sample_distance'])
        self._wavelength = energy2wavelength(float(g_cfg['photon_energy']))
        self._update_moving_average(g_cfg)

        cfg = self._meta.hget_all(mt.AZIMUTHAL_INTEG_PROC)
        self._pixel1 = float(cfg['pixel_size_y'])
        self._pixel2 = float(cfg['pixel_size_x'])
        self._poni1 = int(cfg['integ_center_y']) * self._pixel1
        self._poni2 = int(cfg['integ_center_x']) * self._pixel2

        self._integ_method = cfg['integ_method']
        self._integ_range = self.str2tuple(cfg['integ_range'])
        self._integ_points = int(cfg['integ_points'])
        self._normalizer = Normalizer(int(cfg['normalizer']))
        self._auc_range = self.str2tuple(cfg['auc_range'])
        self._fom_integ_range = self.str2tuple(cfg['fom_integ_range'])

    def _update_integrator(self):
        if self._integrator is None:
            self._integrator = AzimuthalIntegrator(
                dist=self._sample_dist,
                pixel1=self._pixel1,
                pixel2=self._pixel2,
                poni1=self._poni1,
                poni2=self._poni2,
                rot1=0,
                rot2=0,
                rot3=0,
                wavelength=self._wavelength)
        else:
            if self._integrator.dist != self._sample_dist \
                    or self._integrator.wavelength != self._wavelength \
                    or self._integrator.poni1 != self._poni1 \
                    or self._integrator.poni2 != self._poni2:
                # dist, poni1, poni2, rot1, rot2, rot3, wavelength
                self._integrator.set_param((self._sample_dist,
                                            self._poni1,
                                            self._poni2,
                                            0,
                                            0,
                                            0,
                                            self._wavelength))

        return self._integrator

    def _update_moving_average(self, v):
        pass


class AzimuthalIntegrationProcessorPulse(_AzimuthalIntegrationProcessorBase):
    """Pulse-resolved azimuthal integration processor."""

    @profiler("Azimuthal Integration Processor (Pulse)")
    def process(self, data):
        if not self._meta.has_analysis(AnalysisType.AZIMUTHAL_INTEG_PULSE):
            return

        processed = data['processed']
        assembled = data['detector']['assembled']

        integrator = self._update_integrator()
        integ1d = functools.partial(integrator.integrate1d,
                                    method=self._integ_method,
                                    radial_range=self._integ_range,
                                    correctSolidAngle=True,
                                    polarization_factor=1,
                                    unit="q_A^-1")
        integ_points = self._integ_points

        threshold_mask = processed.image.threshold_mask

        def _integrate1d_imp(i):
            masked = mask_image(assembled[i], threshold_mask=threshold_mask)
            return integ1d(masked, integ_points)

        intensities = []  # pulsed A.I.
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i, ret in zip(range(len(assembled)),
                              executor.map(_integrate1d_imp,
                                           range(len(assembled)))):
                if i == 0:
                    momentum = ret.radial
                intensities.append(ret.intensity)

        # intensities = self._normalize_vfom(
        #     processed, np.array(intensities), self._normalizer,
        #     x=momentum, auc_range=self._auc_range)

        # calculate the difference between each pulse and the
        # first one
        diffs = [p - intensities[0] for p in intensities]

        # calculate the figure of merit for each pulse
        foms = []
        for diff in diffs:
            fom = slice_curve(
                diff, momentum, *self._fom_integ_range)[0]
            foms.append(np.sum(np.abs(fom)))

        processed.pulse.ai.x = momentum
        processed.pulse.ai.vfom = intensities
        processed.pulse.ai.fom = foms
        # Note: It is not correct to calculate the mean of intensities
        #       since the result is equivalent to setting all nan to zero
        #       instead of nanmean.


class AzimuthalIntegrationProcessorTrain(_AzimuthalIntegrationProcessorBase):
    """Train-resolved azimuthal integration processor."""

    _intensity_ma = MovingAverageArray()
    _intensity_on_ma = MovingAverageArray()
    _intensity_off_ma = MovingAverageArray()

    def __init__(self):
        super().__init__()

        self._ma_window = 1

    def _update_moving_average(self, cfg):
        if 'reset_ma_ai' in cfg:
            # reset moving average
            del self._intensity_ma
            del self._intensity_on_ma
            del self._intensity_off_ma
            self._meta.hdel(mt.GLOBAL_PROC, 'reset_ma_ai')

        v = int(cfg['ma_window'])
        if self._ma_window != v:
            self.__class__._intensity_ma.window = v
            self.__class__._intensity_on_ma.window = v
            self.__class__._intensity_off_ma.window = v

        self._ma_window = v

    @profiler("Azimuthal Integration Processor (Train)")
    def process(self, data):
        processed = data['processed']

        integrator = self._update_integrator()
        integ1d = functools.partial(integrator.integrate1d,
                                    method=self._integ_method,
                                    radial_range=self._integ_range,
                                    correctSolidAngle=True,
                                    polarization_factor=1,
                                    unit="q_A^-1")
        integ_points = self._integ_points

        if self._meta.has_analysis(AnalysisType.AZIMUTHAL_INTEG):
            mean_ret = integ1d(processed.image.masked_mean, integ_points)

            momentum = mean_ret.radial
            intensity = self._normalize_vfom(
                processed, mean_ret.intensity, self._normalizer,
                x=momentum, auc_range=self._auc_range)

            self._intensity_ma = intensity

            fom = slice_curve(self._intensity_ma, momentum, *self._fom_integ_range)[0]
            fom = np.sum(np.abs(fom))

            processed.ai.x = momentum
            processed.ai.vfom = self._intensity_ma
            processed.ai.fom = fom

        # ------------------------------------
        # pump-probe azimuthal integration
        # ------------------------------------

        if processed.pp.analysis_type == AnalysisType.AZIMUTHAL_INTEG:
            pp = processed.pp

            on_image = pp.image_on
            off_image = pp.image_off

            if on_image is not None and off_image is not None:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    on_off_rets = executor.map(
                        lambda x: integ1d(*x), ((on_image, integ_points),
                                                (off_image, integ_points)))
                on_ret, off_ret = on_off_rets

                momentum = on_ret.radial

                self._intensity_on_ma = on_ret.intensity
                self._intensity_off_ma = off_ret.intensity

                vfom_on, vfom_off = self._normalize_vfom_pp(
                    processed, self._intensity_on_ma, self._intensity_off_ma,
                    self._normalizer, x=on_ret.radial, auc_range=self._auc_range)

                vfom = vfom_on - vfom_off
                sliced = slice_curve(vfom, momentum, *self._fom_integ_range)[0]

                if pp.abs_difference:
                    fom = np.sum(np.abs(sliced))
                else:
                    fom = np.sum(sliced)

                pp.vfom_on = vfom_on
                pp.vfom_off = vfom_off
                pp.vfom = vfom
                pp.x = momentum
                pp.fom = fom
                pp.x_label = processed.ai.x_label
                pp.vfom_label = f"[pump-probe] {processed.ai.vfom_label}"