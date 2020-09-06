"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QColor
from PyQt5.QtWidgets import QPushButton, QSplitter, QLCDNumber
from PyQt5.QtCore import Qt, pyqtSlot

from extra_foam.gui.ctrl_widgets.smart_widgets import (
    SmartBoundaryLineEdit, SmartLineEdit, SmartStringLineEdit
)
from extra_foam.gui.ctrl_widgets import MaskCtrlWidget

from extra_foam.gui.misc_widgets import FColor
from extra_foam.gui.plot_widgets import (
    ImageViewF, ImageAnalysis, TimedImageViewF, TimedPlotWidgetF
)
from extra_foam.config import MaskState

from extra_foam.special_suite.special_analysis_base import (
    create_special, QThreadFoamClient, _BaseAnalysisCtrlWidgetS,
    _SpecialAnalysisBase
)

from .fth_proc import (
    FTHProcessor, _DEFAULT_N_BINS, _DEFAULT_BIN_RANGE
)

_MAX_N_BINS = 999


class FTHCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """FTH analysis control widget.

    """

    def __init__(self, topic, **kwargs):
        super().__init__(topic, **kwargs)

        self.window = SmartLineEdit(str(200))
        self.window.setValidator(QIntValidator(1, 10000))

        self.mask = MaskCtrlWidget()
        self.updateMask_btn = QPushButton('Update Mask', self)
        #self._ntrains = QLCDNumber(5)

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""

        #self._setLcdStyle(self._ntrains)

        layout = self.layout()
        
        layout.addRow("Average window", self.window)
        layout.addRow(self.mask)
        layout.addRow(self.updateMask_btn)
        #layout.addRow(self._ntrains)

    def initConnections(self):
        """Override."""
        #self.swap_btn.clicked.connect(self._swapDataSources)

    def _setLcdStyle(self, lcd):
        lcd.setLineWidth(0)
        lcd.setSegmentStyle(QLCDNumber.Flat)
        palette = lcd.palette()
        palette.setColor(palette.WindowText, QColor(85, 85, 255))
        lcd.setPalette(palette)

class FTHImageView(ImageViewF):
    """FTHImageView class.

    Visualize ROIs.
    """
    def __init__(self, key, **kwargs):
        """Initialization."""

        self._key = key
        super().__init__(has_roi=False, **kwargs)

        self.setTitle(self._key)

    def updateF(self, data):
        """Override."""
        self.setImage(data[self._key])
        if self._key == 'image':
            self.setTitle(f"image <#{data['counts']}>")
        

class FTHImageMaskView(ImageAnalysis):
    """FTHImageView class.

    Visualize ROIs.
    """
    def __init__(self, key, **kwargs):
        """Initialization."""
        super().__init__(has_roi=True, **kwargs)
        self._key = key
        self.setTitle(self._key)
        self.img = None

    def updateF(self, data):
        """Override."""
        self.setImage(data['fth_masked'])
        self.update_mask_f = data['update_mask_f']

    @pyqtSlot(bool)
    def onDrawMask(self, state):
        self.setMaskingState(MaskState.MASK, state)

    @pyqtSlot(bool)
    def onEraseMask(self, state):
        self.setMaskingState(MaskState.UNMASK, state)

    @pyqtSlot()
    def onLoadMask(self):
        self.loadImageMask()

    @pyqtSlot()
    def onSaveMask(self):
        self.saveImageMask()

    @pyqtSlot()
    def onRemoveMask(self):
        self.removeMask()

    @pyqtSlot(bool)
    def onMaskSaveInModulesChange(self, state):
        self.setMaskSaveInModules(state)

    def updateMask(self):
        rect = self._mask_item._selectedRect()
        iserase = self._mask_item.state == MaskState.UNMASK
        self.update_mask_f(int(rect.x()),
                int(rect.y()),
                int(rect.width()),
                int(rect.height()), iserase)
        
@create_special(FTHCtrlWidget, FTHProcessor, QThreadFoamClient)
class FTHWindow(_SpecialAnalysisBase):
    """Main GUI for FTH analysis."""

    icon = "FTH.png"
    _title = "FTH"
    _long_title = "FTH"

    def __init__(self, topic):
        """Initialization."""
        super().__init__(topic)

        self._image = FTHImageView(parent=self, key='image')
        self._fth = FTHImageMaskView(parent=self, key='fth')
        
       
        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""

        main_panel = QSplitter(Qt.Horizontal)
        main_panel.addWidget(self._image)
        main_panel.addWidget(self._fth)
        
        cw = self.centralWidget()
        cw.addWidget(main_panel)
 
        self.resize(self._TOTAL_W, self._TOTAL_H)

    def initConnections(self):
        """Override."""

        self._ctrl_widget_st.window.value_changed_sgn.connect(
                self._worker_st.onWindowChanged)
        self._ctrl_widget_st.window.returnPressed.emit()

        self._ctrl_widget_st.mask.draw_mask_btn.toggled.connect(
                self._fth.onDrawMask)
        self._ctrl_widget_st.mask.erase_mask_btn.toggled.connect(
                self._fth.onEraseMask)
        self._ctrl_widget_st.mask.remove_btn.clicked.connect(
                self._fth.onRemoveMask)
        self._ctrl_widget_st.mask.load_btn.clicked.connect(
                self._fth.onLoadMask)
        self._ctrl_widget_st.mask.save_btn.clicked.connect(
                self._fth.onSaveMask)
        self._ctrl_widget_st.mask.mask_save_in_modules_cb.toggled.connect(
                self._fth.onMaskSaveInModulesChange)
        self._ctrl_widget_st.updateMask_btn.clicked.connect(
                self._fth.updateMask)
