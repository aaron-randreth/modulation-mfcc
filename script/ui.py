from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

import pyqtgraph as pg

from calc import MinMaxFinder

def create_plot_widget(x, y, color='r'):
    plot = pg.PlotWidget()
    plot.plot(x=x, y=y, pen=color)
    return plot

class SelectableListDialog(QtWidgets.QDialog):
    def __init__(self, num_items: int, format_string: str):
        super().__init__()
        self.setWindowTitle('Selectable List')
        self.item_labels = [format_string.format(i) for i in range(num_items)]
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.list_widget.addItems(self.item_labels)
        self.dialog_buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.dialog_buttons.accepted.connect(self.accept)
        self.dialog_buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.list_widget)
        layout.addWidget(self.dialog_buttons)
        self.setLayout(layout)

    def get_selected_indices(self) -> list[int]:
        selected_texts = [item.text() for item in self.list_widget.selectedItems()]
        return [self.item_labels.index(text) for text in selected_texts]

class Crosshair:
    def __init__(self, central_plots) -> None:
        self.central_plots = []
        self.display_plots = []
        self.crosshair_lines = []
        for plot in central_plots:
            self.add_central_plot(plot)
        self.link_plots()

    @property
    def plots(self):
        return [*self.central_plots, *self.display_plots]

    def link_plots(self):
        for p in self.plots:
            p.setXLink(self.central_plots[0])

    def add_central_plot(self, central_plot) -> None:
        line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(style=Qt.DashLine, color="r")
        )
        self.crosshair_lines.append(line)
        self.central_plots.append(central_plot)
        central_plot.addItem(line, ignoreBounds=True)
        central_plot.scene().sigMouseMoved.connect(self.move_crosshair)
        self.link_plots()

    def add_display_plot(self, display_plot) -> None:
        line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(style=Qt.DashLine, color="b")
        )
        self.crosshair_lines.append(line)
        self.display_plots.append(display_plot)
        display_plot.addItem(line, ignoreBounds=True)
        self.link_plots()

    def move_crosshair(self, event):
        mousePoint = None
        pos = event
        for p in self.central_plots:
            if p.sceneBoundingRect().contains(pos):
                mousePoint = p.getPlotItem().vb.mapSceneToView(pos)
        if mousePoint is None:
            return
        for l in self.crosshair_lines:
            l.setPos(mousePoint.x())

    def add_panel_plot(self, panel_plot):
        line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(style=Qt.DashLine, color="g")
        )
        self.crosshair_lines.append(line)
        self.central_plots.append(panel_plot)
        panel_plot.addItem(line, ignoreBounds=True)
        panel_plot.scene().sigMouseMoved.connect(self.move_crosshair)
        self.link_plots()

class MinMaxAnalyser(QtWidgets.QWidget):
    def __init__(self, name: str, x, y, extremum: MinMaxFinder, get_interval_func, color='r', secondary_viewbox=None, tertiary_viewbox=None) -> None:
        super().__init__()
        self.name = name
        self.x = x
        self.y = y
        self.extremum = extremum
        self.get_interval = get_interval_func
        self.color = color
        self.secondary_viewbox = secondary_viewbox
        self.tertiary_viewbox = tertiary_viewbox
        self.plot_widget = None
        self.visibility_checkbox = QtWidgets.QCheckBox(f"Toggle visibility for {name}")
        self.visibility_checkbox.setChecked(True)
        self.__init_ui()
        self.max_points = pg.ScatterPlotItem(pen=pg.mkPen("g"), brush=pg.mkBrush("b"))
        self.min_points = pg.ScatterPlotItem(pen=pg.mkPen("r"), brush=pg.mkBrush("r"))
        self.plot_widget.addItem(self.max_points)
        self.plot_widget.addItem(self.min_points)
        self.max_points.hide()
        self.min_points.hide()

    def __init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout()
        self.toolbar = QtWidgets.QToolBar()
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setMaximumHeight(400)
        
        # Utilisez ScatterPlotItem pour rendre les points cliquables
        self.curve = pg.ScatterPlotItem(x=self.x, y=self.y, pen=self.color, brush=pg.mkBrush(self.color))
        self.curve.sigClicked.connect(self.add_point_on_click)
        self.plot_widget.addItem(self.curve)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    def update_plot(self, x, y):
        self.curve.setData(x=x, y=y)

    def add_point_on_click(self, plot_item, event):
        pos = event[0].scenePos()
        if not plot_item.getViewBox().sceneBoundingRect().contains(pos):
            return

        mouse_point = plot_item.getViewBox().mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        print(f"Clicked at x: {x}, y: {y}")  # Affiche les coordonnées du clic

        if self.parent().manual_peak_maximum_addition.isChecked():
            points_x, points_y = self.max_points.getData()
            closest_index = np.argmin(np.abs(points_x - x))
            points_x = np.insert(points_x, closest_index, x)
            points_y = np.insert(points_y, closest_index, y)
            self.max_points.setData(points_x, points_y)
            self.max_points.show()
            print("Added max point")
        elif self.parent().manual_peak_minimum_addition.isChecked():
            points_x, points_y = self.min_points.getData()
            closest_index = np.argmin(np.abs(points_x - x))
            points_x = np.insert(points_x, closest_index, x)
            points_y = np.insert(points_y, closest_index, y)
            self.min_points.setData(points_x, points_y)
            self.min_points.show()
            print("Added min point")
        elif self.parent().manual_peak_removal.isChecked():
            points_x, points_y = self.max_points.getData()
            distances = np.sqrt((points_x - x) ** 2 + (points_y - y) ** 2)
            closest_index = np.argmin(distances)
            points_x = np.delete(points_x, closest_index)
            points_y = np.delete(points_y, closest_index)
            self.max_points.setData(points_x, points_y)
            self.max_points.show()
            print("Removed point")
