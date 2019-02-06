"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

ImageToolWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..widgets.pyqtgraph import QtCore, QtGui

from ..widgets import ImageView
from .base_window import AbstractWindow, SingletonWindow
from ..config import config


class RoiCtrlWidget(QtGui.QGroupBox):
    """Widget for controlling of an ROI."""
    # activated, w, h, cx, cy
    roi_region_changed_sgn = QtCore.Signal(bool, int, int, int, int)

    _pos_validator = QtGui.QIntValidator(-10000, 10000)
    _size_validator = QtGui.QIntValidator(0, 10000)

    def __init__(self, roi, *, title="ROI control", parent=None):
        """Initialization.

        :param RectROI roi: RectROI object.
        """
        super().__init__(title, parent=parent)
        self._roi = roi

        self._width_le = QtGui.QLineEdit()
        self._width_le.setValidator(self._size_validator)
        self._height_le = QtGui.QLineEdit()
        self._height_le.setValidator(self._size_validator)
        self._cx_le = QtGui.QLineEdit()
        self._cx_le.setValidator(self._pos_validator)
        self._cy_le = QtGui.QLineEdit()
        self._cy_le.setValidator(self._pos_validator)
        self._width_le.editingFinished.connect(self.roiRegionChangedEvent)
        self._height_le.editingFinished.connect(self.roiRegionChangedEvent)
        self._cx_le.editingFinished.connect(self.roiRegionChangedEvent)
        self._cy_le.editingFinished.connect(self.roiRegionChangedEvent)

        self._line_edits = (self._width_le, self._height_le,
                            self._cx_le, self._cy_le)

        self.activate_cb = QtGui.QCheckBox("Activate")
        self.lock_cb = QtGui.QCheckBox("Lock")
        self.lock_aspect_cb = QtGui.QCheckBox("Lock aspect ratio")

        self.initUI()

        roi.sigRegionChangeFinished.connect(self.onRoiRegionChangeFinished)
        roi.sigRegionChangeFinished.emit(roi)  # fill the QLineEdit(s)
        self.activate_cb.stateChanged.connect(self.toggleRoiActivationEvent)
        self.lock_cb.stateChanged.connect(self.lockEvent)
        self.lock_aspect_cb.stateChanged.connect(self.lockAspectEvent)

    def initUI(self):
        le_layout = QtGui.QHBoxLayout()
        le_layout.addWidget(QtGui.QLabel("Width: "))
        le_layout.addWidget(self._width_le)
        le_layout.addWidget(QtGui.QLabel("Height: "))
        le_layout.addWidget(self._height_le)
        le_layout.addWidget(QtGui.QLabel("X center: "))
        le_layout.addWidget(self._cx_le)
        le_layout.addWidget(QtGui.QLabel("Y center: "))
        le_layout.addWidget(self._cy_le)

        cb_layout = QtGui.QHBoxLayout()
        cb_layout.addWidget(self.activate_cb)
        cb_layout.addWidget(self.lock_cb)
        cb_layout.addWidget(self.lock_aspect_cb)

        layout = QtGui.QVBoxLayout()

        layout.addLayout(cb_layout)
        layout.addLayout(le_layout)

        self.setLayout(layout)

    @QtCore.pyqtSlot(object)
    def onRoiRegionChangeFinished(self, roi):
        """Connect to the signal from an ROI object."""
        w, h = [int(v) for v in roi.size()]
        cx, cy = [int(v) for v in roi.pos()]
        self.updateParameters(w, h, cx, cy)
        # inform widgets outside this window
        self.roi_region_changed_sgn.emit(True, w, h, cx, cy)

    def toggleRoiActivationEvent(self, state):
        if state == QtCore.Qt.Checked:
            self._roi.show()
            self.enableAllEdit()
            self.roi_region_changed_sgn.emit(
                True, *self._roi.size(), *self._roi.pos())
        else:
            self._roi.hide()
            self.disableAllEdit()
            self.roi_region_changed_sgn.emit(
                False, *self._roi.size(), *self._roi.pos())

    def lockEvent(self, state):
        if state == QtCore.Qt.Checked:
            self._roi.lock()
            self.disableLockEdit()
        else:
            self._roi.unLock()
            self.enableLockEdit()

    def lockAspectEvent(self, state):
        if state == QtCore.Qt.Checked:
            self._roi.lockAspect()
        else:
            self._roi.unLockAspect()

    def updateParameters(self, w, h, cx, cy):
        self._width_le.setText(str(w))
        self._height_le.setText(str(h))
        self._cx_le.setText(str(cx))
        self._cy_le.setText(str(cy))

    def roiRegionChangedEvent(self):
        w = int(self._width_le.text())
        h = int(self._height_le.text())
        cx = int(self._cx_le.text())
        cy = int(self._cy_le.text())

        # If 'update' == False, the state change will be remembered
        # but not processed and no signals will be emitted.
        self._roi.setSize((w, h), update=False)
        self._roi.setPos((cx, cy), update=False)

        self.roi_region_changed_sgn.emit(True, w, h, cx, cy)

    def disableLockEdit(self):
        for w in self._line_edits:
            w.setDisabled(True)
        self.lock_aspect_cb.setDisabled(True)

    def enableLockEdit(self):
        for w in self._line_edits:
            w.setDisabled(False)
        self.lock_aspect_cb.setDisabled(False)

    def disableAllEdit(self):
        self.disableLockEdit()
        self.lock_cb.setDisabled(True)

    def enableAllEdit(self):
        self.lock_cb.setDisabled(False)
        self.enableLockEdit()


class MaskCtrlWidget(QtGui.QGroupBox):
    """Widget for masking image."""

    threshold_mask_sgn = QtCore.pyqtSignal(float, float)

    _double_validator = QtGui.QDoubleValidator()

    def __init__(self, title, *, parent=None):
        """"""
        super().__init__(title, parent=parent)

        self._min_pixel_le = QtGui.QLineEdit(str(config["MASK_RANGE"][0]))
        self._min_pixel_le.setValidator(self._double_validator)
        self._max_pixel_le = QtGui.QLineEdit(str(config["MASK_RANGE"][1]))
        self._max_pixel_le.setValidator(self._double_validator)
        self._min_pixel_le.returnPressed.connect(
            self.thresholdMaskChangedEvent)
        self._max_pixel_le.returnPressed.connect(
            self.thresholdMaskChangedEvent)

        self.initUI()

    def initUI(self):
        threshold_layout = QtGui.QHBoxLayout()
        threshold_layout.addWidget(QtGui.QLabel("Min.: "))
        threshold_layout.addWidget(self._min_pixel_le)
        threshold_layout.addWidget(QtGui.QLabel("Max.: "))
        threshold_layout.addWidget(self._max_pixel_le)

        layout = QtGui.QVBoxLayout()

        layout.addLayout(threshold_layout)
        self.setLayout(layout)

    def thresholdMaskChangedEvent(self):
        self.threshold_mask_sgn.emit(float(self._min_pixel_le.text()),
                                     float(self._max_pixel_le.text()))


@SingletonWindow
class ImageToolWindow(AbstractWindow):
    """ImageToolWindow class.

    This tool provides selecting of ROI and image masking.
    """
    title = "Image tool"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._image_view = ImageView(lock_roi=False)

        self._clear_roi_hist_btn = QtGui.QPushButton("Clear ROI history")
        self._clear_roi_hist_btn.clicked.connect(
            self._mediator.onRoiHistClear)

        self._roi_hist_window_le = QtGui.QLineEdit(str(600))
        validator = QtGui.QIntValidator()
        validator.setBottom(1)
        self._roi_hist_window_le.setValidator(validator)
        self._roi_hist_window_le.editingFinished.connect(
            self._mediator.onRoiIntensityWindowChange
        )

        self._roi1_ctrl = RoiCtrlWidget(
            self._image_view.roi1,
            title="ROI 1 ({})".format(config['ROI_COLORS'][0]))
        self._roi2_ctrl = RoiCtrlWidget(
            self._image_view.roi2,
            title="ROI 2 ({})".format(config['ROI_COLORS'][1]))

        self._roi1_ctrl.roi_region_changed_sgn.connect(
            self._mediator.onRoi1Changed)
        self._roi2_ctrl.roi_region_changed_sgn.connect(
            self._mediator.onRoi2Changed)

        self._mask_panel = MaskCtrlWidget("Masking tool")
        self._mask_panel.threshold_mask_sgn.connect(
            self._mediator.onThresholdMaskChange)

        self._update_image_btn = QtGui.QPushButton("Update image")
        self._update_image_btn.clicked.connect(self.updateImage)

        self.initUI()
        self.resize(800, 800)
        self.updateImage()

    def initUI(self):
        """Override."""
        roi_ctrl_layout = QtGui.QHBoxLayout()
        roi_ctrl_layout.addWidget(QtGui.QLabel("ROI monitor window size: "))
        roi_ctrl_layout.addWidget(self._roi_hist_window_le)
        roi_ctrl_layout.addWidget(self._clear_roi_hist_btn)

        tool_layout = QtGui.QVBoxLayout()
        tool_layout.addLayout(roi_ctrl_layout)
        tool_layout.addWidget(self._roi1_ctrl)
        tool_layout.addWidget(self._roi2_ctrl)

        layout = QtGui.QGridLayout()
        layout.addWidget(self._image_view, 0, 0, 1, 1)
        layout.addLayout(tool_layout, 1, 0, 1, 1)
        layout.addWidget(self._mask_panel, 0, 1, 1, 1)
        layout.addWidget(self._update_image_btn, 1, 1, 1, 1)

        self._cw.setLayout(layout)

    def updateImage(self):
        """For updating image manually."""
        data = self._data.get()
        if data.empty():
            return

        self._image_view.setImage(data.image_mean)
