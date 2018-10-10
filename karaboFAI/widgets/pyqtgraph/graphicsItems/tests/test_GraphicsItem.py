import gc
import weakref
try:
    import faulthandler
    faulthandler.enable()
except ImportError:
    pass

from widgets import pyqtgraph as pg

pg.mkQApp()

def test_getViewWidget():
    view = pg.PlotWidget()
    vref = weakref.ref(view)
    item = pg.InfiniteLine()
    view.addItem(item)
    assert item.getViewWidget() is view
    del view
    gc.collect()
    assert vref() is None
    assert item.getViewWidget() is None

def test_getViewWidget_deleted():
    view = pg.PlotWidget()
    item = pg.InfiniteLine()
    view.addItem(item)
    assert item.getViewWidget() is view
    
    # Arrange to have Qt automatically delete the view widget
    obj = pg.QtGui.QWidget()
    view.setParent(obj)
    del obj
    gc.collect()

    assert not widgets.pyqtgraph.Qt.isQObjectAlive(view)
    assert item.getViewWidget() is None