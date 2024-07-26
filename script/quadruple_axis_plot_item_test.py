import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyqtgraph as pg

# Assuming QuadrupleAxisPlotItem is defined in a file named quadruple_axis_plot_item.py
from quadruple_axis_plot_item import QuadrupleAxisPlotItem, Panel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI2()

    def initUI1(self):
        self.setWindowTitle('Quadruple Axis Plot Example')
        self.setGeometry(100, 100, 800, 600)

        # Create a PlotWidget with a QuadrupleAxisPlotItem
        plot_widget = pg.PlotWidget()
        self.setCentralWidget(plot_widget)

        plot_item = QuadrupleAxisPlotItem()
        plot_widget.setCentralItem(plot_item)
        plot_item.add_viewboxes_to_scene()

        # Generate some sample data
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        y3 = np.tan(x)
        y4 = np.exp(x)

        # Add data to the main axis
        curve1 = pg.PlotCurveItem(x, y1, pen='w')
        curve2 = pg.PlotCurveItem(x, y2, pen='r')
        curve3 = pg.PlotCurveItem(x, y3, pen='g')
        curve4 = pg.PlotCurveItem(x, y4, pen='b')

        plot_item.add_item("left", curve1)
        # plot_item.remove_item("left", curve1)

        # Add data to the second axis (right)
        plot_item.add_item("right", curve2)
        # plot_item.remove_item("right", curve2)

        # Add data to the third axis (left2)
        plot_item.add_item("left_bis", curve3)
        # plot_item.remove_item("left_bis", curve3)
        
        # Add data to the fourth axis (right2)
        plot_item.add_item("right_bis", curve4)

    def initUI2(self):
        self.setWindowTitle('Quadruple Axis Plot Example')
        self.setGeometry(100, 100, 800, 600)

        # Create a PlotWidget with a QuadrupleAxisPlotItem
        plot_widget = pg.PlotWidget()
        self.setCentralWidget(plot_widget)

        plot_item = Panel()
        plot_widget.setCentralItem(plot_item)
        plot_item.add_viewboxes_to_scene()

        # Generate some sample data
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        y3 = np.tan(x)
        y4 = np.exp(x)
        y5 = np.cos(x) + np.sin(x)

        # Add data to the main axis
        curve1 = pg.PlotCurveItem(x, y1, pen='w')
        curve2 = pg.PlotCurveItem(x, y2, pen='r')
        curve3 = pg.PlotCurveItem(x, y3, pen='g')
        curve4 = pg.PlotCurveItem(x, y4, pen='b')
        curve5 = pg.PlotCurveItem(x, y5, pen='r')

        plot_item.add_item(curve1)
        plot_item.add_item(curve2)
        plot_item.add_item(curve3)
        plot_item.add_item(curve4)

        plot_item.remove_item(curve2)
        plot_item.remove_item(curve4)

        plot_item.add_item(curve5)

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

