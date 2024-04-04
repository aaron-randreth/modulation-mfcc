from time import sleep
import os
import sys

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout,
        QMainWindow, QApplication,
        QLabel,
    )

import pyqtgraph as pg
import tgt

from markers import (
        Marker, IntervalMarker,
)

from tiers import (
        TextGrid,
        PointTier,
        IntervalTier,
)

from textgridtools import (
        TextgridConverter,
        PointTierTGTConvert,
        IntervalTierTGTConvert,
        TextgridTGTConvert,
)


def init_linked_plot():
    linked_plot = pg.PlotWidget()

    xmin = 0
    xmax = 2.9

    linked_plot.setXRange(xmin, xmax)
    linked_plot.setLimits(xMin=xmin, xMax=xmax)
    
    return linked_plot, xmin, xmax



def init_tgt():

    linked_plot, xmin, xmax = init_linked_plot()
    textgrid = TextGrid(linked_plot, TextgridTGTConvert())

    pt = PointTier("Bob", xmin, xmax, PointTierTGTConvert())
    it = IntervalTier("Itar", xmin, xmax, IntervalTierTGTConvert())

    textgrid.add_tier(pt)
    textgrid.add_tier(it)

    pt.add_element(Marker(1, "DOOOOPE"))
    pt.add_element(Marker(2))
    pt.add_element(Marker(3))
    # pt.remove_element_by_idx(1)

    it.add_element(IntervalMarker.new_interval(0, 1, "DUm"))
    # it.add_element(IntervalMarker.new_interval(0.5, 2, "DOUm"))
    it.add_element(IntervalMarker.new_interval(1, 2, "DOUm"))

    return textgrid




# Snip...


# Step 1: Create a worker class
class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    tgt = None

    def __init__(self, tgt):
        super().__init__()
        self.tgt = tgt

    def run(self):
        while True:
            print(tgt.io.export_to_long_textgrid((self.tgt.to_textgrid())))
            sleep(10)
            os.system("clear")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.runLongTask()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        text_widget = QLabel("This is the firts widget")
        layout.addWidget(text_widget)

        # self.tgt = tgt.io.read_textgrid("dom.TextGrid")
        # linked_plot, _,_ = init_linked_plot()
        # self.tgt = TextgridTGTConvert().from_textgrid(self.tgt, linked_plot)
        
        self.tgt = init_tgt()
        layout.addWidget(self.tgt)

    def runLongTask(self):
        # Step 2: Create a QThread object
        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = Worker(self.tgt)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        # Step 6: Start the thread
        self.thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
