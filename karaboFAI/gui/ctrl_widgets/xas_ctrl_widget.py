"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

XasCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph import QtCore, QtGui

from .base_ctrl_widgets import AbstractCtrlWidget


class XasCtrlWidget(AbstractCtrlWidget):
    """Analysis parameters setup for pump-probe experiments."""

    def __init__(self, *args, **kwargs):
        super().__init__("XAS analysis setup", *args, **kwargs)

        self._reset_btn = QtGui.QPushButton("Reset")

        self._nbins_le = QtGui.QLineEdit("60")
        self._nbins_le.setValidator(QtGui.QIntValidator(0, 999))

        self.initUI()

        self.setFixedHeight(self.minimumSizeHint().height())

        self.initConnections()

    def initUI(self):
        """Overload."""
        layout = QtGui.QGridLayout()
        AR = QtCore.Qt.AlignRight

        layout.addWidget(QtGui.QLabel("# of energy bins: "), 0, 0, 1, 1, AR)
        layout.addWidget(self._nbins_le, 0, 1, 1, 1)
        layout.addWidget(self._reset_btn, 0, 2, 1, 2)

        self.setLayout(layout)

    def initConnections(self):
        mediator = self._mediator

        self._reset_btn.clicked.connect(mediator.xas_state_set_sgn)

        self._nbins_le.editingFinished.connect(
            lambda: mediator.xas_energy_bins_change_sgn.emit(
                int(self._nbins_le.text())))
        self._nbins_le.editingFinished.emit()
