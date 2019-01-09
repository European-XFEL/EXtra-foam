from h5py import File
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from karabo_data.geometry import LPDGeometry
# from karabo_data.geometry2 import AGIPD_1MGeometry

from ..config import config
from ..widgets.pyqtgraph import QtGui, QtCore


class GeometryLayout(QtGui.QWidget):
    """Geometry layout widget

    TODO: Create a widget that will be used to display geometry layout
    in the control panel.
    How to update. Not working
    """

    def __init__(self, size=(5.0, 4.0), dpi=100, parent=None):
        super().__init__(parent=parent)
        parent.geometry_sgn.connect(self.onGeometryChanged)
        parent.updateSharedParameters()
        self.geom_sp = None

        self.fig = None

        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    @QtCore.pyqtSlot(str, list)
    def onGeometryChanged(self, filename, quad_positions):
        if config['TOPIC'] == 'FXE':
            with File(filename, 'r') as f:
                self.geom_sp = LPDGeometry.from_h5_file_and_quad_positions(
                    f, quad_positions)
        elif config['TOPIC'] == 'SPB':
            self.geom_sp = AGIPD_1MGeometry.from_crystfel_geom(filename).snap()

        self.fig = self.geom_sp.inspect()
        self.canvas = FigureCanvas(self.fig)
        self.draw()

    def draw(self):
        self.canvas.draw()
