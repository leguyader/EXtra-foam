"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

RoiWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph.dockarea import Dock

from .base_window import DockerWindow
from ..plot_widgets import RoiImageView
from ...config import config


class RoiWindow(DockerWindow):
    """RoiWindow class."""
    title = "ROI"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    _LW1 = 0.5 * _TOTAL_W
    _LW2 = _TOTAL_W
    _LH1 = 0.35 * _TOTAL_H
    _LH2 = 0.3 * _TOTAL_H

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._roi1_image = RoiImageView(1, parent=self)
        self._roi2_image = RoiImageView(2, parent=self)
        self._roi3_image = RoiImageView(3, parent=self)
        self._roi4_image = RoiImageView(4, parent=self)

        self.initUI()
        self.initConnections()

        self.resize(self._TOTAL_W, self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""
        roi3_image_dock = Dock("ROI3", size=(self._LW1, self._LH1))
        self._docker_area.addDock(roi3_image_dock)
        roi3_image_dock.addWidget(self._roi3_image)

        roi4_image_dock = Dock("ROI4", size=(self._LW1, self._LH1))
        self._docker_area.addDock(roi4_image_dock, 'right', roi3_image_dock)
        roi4_image_dock.addWidget(self._roi4_image)

        roi1_image_dock = Dock("ROI1", size=(self._LW1, self._LH1))
        self._docker_area.addDock(roi1_image_dock, 'above', roi3_image_dock)
        roi1_image_dock.addWidget(self._roi1_image)

        roi2_image_dock = Dock("ROI2", size=(self._LW1, self._LH1))
        self._docker_area.addDock(roi2_image_dock, 'above', roi4_image_dock)
        roi2_image_dock.addWidget(self._roi2_image)

    def initConnections(self):
        """Override."""
        pass