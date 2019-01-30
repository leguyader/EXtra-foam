"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

AnalysisCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_ctrl_widgets import AbstractCtrlWidget
from ..config import config
from ..helpers import parse_boundary
from ..logger import logger
from ..widgets.pyqtgraph import QtCore, QtGui


class AnalysisCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the general analysis parameters."""

    image_mask_range_sgn = QtCore.pyqtSignal(float, float)

    pulse_id_range_sgn = QtCore.pyqtSignal(int, int)
    vip_pulse_id1_sgn = QtCore.pyqtSignal(int)
    vip_pulse_id2_sgn = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__("General analysis setup", *args, **kwargs)

        # We keep the definitions of attributes which are not used in the
        # PULSE_RESOLVED = True case. It makes sense since these attributes
        # also appear in the defined methods.

        if self._pulse_resolved:
            min_pulse_id = 0
            max_pulse_id = 2700
            vip_pulse_id1 = 0
            vip_pulse_id2 = 1
        else:
            min_pulse_id = 0
            max_pulse_id = 1  # not included, Python convention
            vip_pulse_id1 = 0
            vip_pulse_id2 = 0

        self._min_pulse_id_le = QtGui.QLineEdit(str(min_pulse_id))
        self._min_pulse_id_le.setEnabled(False)
        self._max_pulse_id_le = QtGui.QLineEdit(str(max_pulse_id))

        self._vip_pulse_id1_le = QtGui.QLineEdit(str(vip_pulse_id1))
        self._vip_pulse_id1_le.returnPressed.connect(
            self.onVipPulse1Confirmed)
        self._vip_pulse_id2_le = QtGui.QLineEdit(str(vip_pulse_id2))
        self._vip_pulse_id2_le.returnPressed.connect(
            self.onVipPulse2Confirmed)

        self._image_mask_range_le = QtGui.QLineEdit(
            ', '.join([str(v) for v in config["MASK_RANGE"]]))

        self._disabled_widgets_during_daq = [
            self._max_pulse_id_le,
            self._image_mask_range_le
        ]

        self.initUI()

    def initUI(self):
        """Overload."""
        layout = QtGui.QFormLayout()
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        layout.addRow("Image mask range: ", self._image_mask_range_le)

        if self._pulse_resolved:
            layout.addRow("Min. pulse ID: ", self._min_pulse_id_le)
            layout.addRow("Max. pulse ID: ", self._max_pulse_id_le)
            layout.addRow("VIP pulse ID 1: ", self._vip_pulse_id1_le)
            layout.addRow("VIP pulse ID 2: ", self._vip_pulse_id2_le)

        self.setLayout(layout)

    def updateSharedParameters(self, log=False):
        """Override"""
        try:
            mask_range = parse_boundary(self._image_mask_range_le.text())
            self.image_mask_range_sgn.emit(*mask_range)
        except ValueError as e:
            logger.error("<Image mask range>: " + str(e))
            return False

        pulse_id_range = (int(self._min_pulse_id_le.text()),
                          int(self._max_pulse_id_le.text()))
        if pulse_id_range[1] <= 0:
            logger.error("<Pulse ID range>: Invalid input!")
            return False
        self.pulse_id_range_sgn.emit(*pulse_id_range)

        self._emit_vip_pulse_id1()
        self._emit_vip_pulse_id2()

        if log:
            if self._pulse_resolved:
                logger.info("<Pulse ID range>: ({}, {})"
                            .format(*pulse_id_range))

        return True

    @QtCore.pyqtSlot()
    def onVipPulse1Confirmed(self):
        self._emit_vip_pulse_id1()

    def _emit_vip_pulse_id1(self):
        try:
            pulse_id = int(self._vip_pulse_id1_le.text().strip())
        except ValueError as e:
            logger.error("<VIP pulse ID 1>: " + str(e))
            return

        if pulse_id < 0:
            logger.error("<VIP pulse ID 1>: pulse ID must be non-negative!")
            return

        self.vip_pulse_id1_sgn.emit(pulse_id)

    @QtCore.pyqtSlot()
    def onVipPulse2Confirmed(self):
        self._emit_vip_pulse_id2()

    def _emit_vip_pulse_id2(self):
        try:
            pulse_id = int(self._vip_pulse_id2_le.text().strip())
        except ValueError as e:
            logger.error("<VIP pulse ID 2>: " + str(e))
            return

        if pulse_id < 0:
            logger.error("<VIP pulse ID 2>: pulse ID must be non-negative!")
            return

        self.vip_pulse_id2_sgn.emit(pulse_id)
