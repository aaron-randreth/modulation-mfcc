import sys
from scipy.signal import argrelextrema
from math import sqrt

from scipy import signal
from librosa import load, feature as lf
import numpy as np
from scipy.signal import find_peaks
import numpy as np
from PyQt5.QtWidgets import QToolBar, QAction
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QGridLayout,
    QLabel,
)

import pyqtgraph as pg

pg.setConfigOptions(foreground="black", background="w")

import tgt
from PyQt5.QtCore import Qt
from praat_py_ui import (
    tiers as ui_tiers,
    textgridtools as ui_tgt,
)

from datasources import datasource, plotter, mfcc, audio_wave, peaks


class AudioAnalyzer(QMainWindow):
    audio_file_path: str
    plot_display: datasource.PlotDisplay

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio Analyzer")
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        layout = QGridLayout(self.main_widget)
        self.layout = layout

        self.audio_file_label = QLabel("Aucun fichier audio chargé")
        layout.addWidget(self.audio_file_label, 0, 0, 1, 3)

        self.init_buttons()
        self.plot_display = None

    def init_buttons(self):
        self.button = QPushButton("Select Audio file")
        self.button.clicked.connect(self.load_audio)
        self.layout.addWidget(self.button, 3, 0)

        # self.update_curves_btn = QPushButton("Reload plots")
        # self.update_curves_btn.clicked.connect()
        # self.layout.addWidget(self.update_curves_btn, 4, 0)

        # self.remove_figure_btn = QPushButton("Remove plots")
        # self.remove_figure_btn.clicked.connect(self.ax.clear)
        # self.layout.addWidget(self.remove_figure_btn, 4, 0)

    def add_plot_display(self, filepath: str):
        self.plot_display = datasource.PlotDisplay("Fichier")

        mfcc_source = mfcc.MfccSource(filepath)

        sources = [
            datasource.DataDisplay(
                "audi-source", "Courbe Audio",
                audio_wave.AudioSource(filepath),
                plotter.TwoDimPlotter()
            ),
            datasource.DataDisplay(
                "mfcc-source", "Modulation mfccc",
                mfcc_source, plotter.TwoDimPlotter(),
                pen="r"

            ),
            datasource.DataDisplay(
                "peak-source", "Modulation mfccc",
                peaks.PeakSource(mfcc_source), plotter.TwoDimPlotter(),
                pen=None, symbol="x"
            ),
        ]

        for s in sources:
            self.plot_display.add_source(s)

        self.layout.addWidget(self.plot_display.principal_plot, 1, 0, 1, 3)

        self.ax = self.plot_display.principal_plot
        self.ax.setLabel("left", "Valeur de modulation")
        self.ax.setLabel("bottom", "Temps", units="s")
        self.ax.setMouseEnabled(y=False)

    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.wav)"
        )
        if not file_path:
            return

        self.audio_file_label.setText(f"Fichier audio chargé : {file_path}")
        self.audio_file_path = file_path

        if self.plot_display is None:
            self.add_plot_display(file_path)

        self.plot_display.change_file(self.audio_file_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = AudioAnalyzer()
    ex.show()
    sys.exit(app.exec_())
