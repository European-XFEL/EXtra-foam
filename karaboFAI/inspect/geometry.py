from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from karabo_data.geometry import LPDGeometry
from karabo_data.geometry2 import AGIPD_1MGeometry

from ..config import config

class GeometryLayout(QtGui.QWidget):
	"""Geometry layout widget

	TODO: Create a widget that will be used to display geometry layout
	in the control panel
	"""

	def __init__(self, size=(10,10)):
		super().__init__(self)

		if config["TOPIC"] == "FXE":
			pass

