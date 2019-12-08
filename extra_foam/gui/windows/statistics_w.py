"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSplitter

from .base_window import _AbstractPlotWindow
from ..plot_widgets import PulsesInTrainFomWidget, FomHistogramWidget
from ...config import config


class StatisticsWindow(_AbstractPlotWindow):
    """StatisticsWindow class.

    Visualize statistics.
    """
    _title = "Statistics"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']
    _TOTAL_W /= 2

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._pulse_fom = PulsesInTrainFomWidget(parent=self)
        self._fom_historgram = FomHistogramWidget(parent=self)

        self.initUI()

        self.resize(self._TOTAL_W, self._TOTAL_H)
        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        self._cw = QSplitter(Qt.Vertical)
        self._cw.addWidget(self._pulse_fom)
        self._cw.addWidget(self._fom_historgram)
        self._cw.setSizes([1, 1])
        self.setCentralWidget(self._cw)

    def initConnections(self):
        """Override."""
        pass