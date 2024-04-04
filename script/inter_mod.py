import sys

from scipy import signal
from librosa import load, feature as lf
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

from praat_py_ui import (
        tiers as ui_tiers,
        textgridtools as ui_tgt,
)


class AudioAnalyzer(QMainWindow):
    textgrid: ui_tiers.TextGrid | None = None

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Analyzer")
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        layout = QGridLayout(self.main_widget)
        self.layout = layout

        self.audio_file_label = QLabel("Aucun fichier audio chargé")
        layout.addWidget(self.audio_file_label, 0, 0, 1, 3)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget, 1, 0, 1, 2)

        self.ax = self.plot_widget.getPlotItem()
        self.ax.setLabel("left", "Valeur de modulation")
        self.ax.setLabel("bottom", "Temps", units="s")
        self.ax.setMouseEnabled(y=False)

        self.init_buttons()

        self.coordinates_widget = QWidget()
        coordinates_layout = QVBoxLayout(self.coordinates_widget)
        layout.addWidget(self.coordinates_widget, 0, 2, 2, 1)

        self.coordinates_label = QLabel("Mesures:")
        coordinates_layout.addWidget(self.coordinates_label)
        self.coordinates_list_label = QLabel()
        coordinates_layout.addWidget(self.coordinates_list_label)
        self.mean_std_label = QLabel()
        coordinates_layout.addWidget(self.mean_std_label)
        self.coordinates_list = []

        self.plot_widget.scene().sigMouseClicked.connect(self.mouse_clicked)

        self.region = pg.LinearRegionItem()
        self.region.setMovable(True)
        self.plot_widget.addItem(self.region)
        self.duration_text = pg.TextItem(anchor=(0.5, 2.5))
        self.plot_widget.addItem(self.duration_text)

        def update_duration():
            region_values = self.region.getRegion()
            duration = region_values[1] - region_values[0]
            self.duration_text.setText(f"Durée sélectionnée : {duration:.2f} s")

        self.region.sigRegionChanged.connect(update_duration)
        update_duration()

        self.tiers = {}

    def init_buttons(self):
        self.button = QPushButton("Load Audio File")
        self.button.clicked.connect(self.load_audio)
        self.layout.addWidget(self.button, 3, 0)

        self.reset_button = QPushButton("Reset Measurements")
        self.reset_button.clicked.connect(self.reset_measurements)
        self.layout.addWidget(self.reset_button, 4, 0)

        self.annotation_button = QPushButton("Load Textgrid Annotation")
        self.annotation_button.clicked.connect(self.load_annotations)
        self.layout.addWidget(self.annotation_button, 5, 0)

        self.annotation_save_button = QPushButton("Save TextGrid Annotation")
        self.annotation_save_button.clicked.connect(self.save_annotations)
        self.layout.addWidget(self.annotation_save_button, 6, 0)

    def mouse_clicked(self, event):
        if self.plot_widget.sceneBoundingRect().contains(event.scenePos()):

            mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(
                event.scenePos()
            )
            x, y = mouse_point.x(), mouse_point.y()

            self.coordinates_list.append((x, y))

            distances = []
            for i in range(1, len(self.coordinates_list)):
                dist = self.coordinates_list[i][0] - self.coordinates_list[i - 1][0]
                distances.append(dist)

            mean_distance = np.mean(distances)

            coordinates_text = "\n".join(
                [
                    f"X: {coord[0]:.2f}, Y: {coord[1]:.2f}"
                    for coord in self.coordinates_list
                ]
            )
            distances_text = "\n".join(
                [f"Distance {i+1}: {distances[i]:.2f}" for i in range(len(distances))]
            )
            self.coordinates_list_label.setText(
                f"Last Coordinate: X: {x:.2f}, Y: {y:.2f}\n{coordinates_text}"
            )
            self.mean_std_label.setText(
                f"Meanch: {np.mean([coord[1] for coord in self.coordinates_list]):.2f}, "
                f"Std Dev of Meanch: {np.std([coord[1] for coord in self.coordinates_list]):.2f}, "
                f"Eventdur : {mean_distance:.2f}\n{distances_text}"
            )

            self.plot_widget.addItem(
                pg.ScatterPlotItem(
                    x=[x],
                    y=[y],
                    symbol="x",
                    size=10,
                    pen=pg.mkPen(None),
                    brush=(255, 255, 255),
                )
            )

    def tier_plot_clicked(self, tier_plot, event):
        if not tier_plot.sceneBoundingRect().contains(event.scenePos()):
            return

        mouse_point = tier_plot.vb.mapSceneToView(event.scenePos())

        x, y = mouse_point.x(), mouse_point.y()
        self.add_interval(x, 1, "test", tier_plot, "phones")

    def reset_measurements(self):
        self.coordinates_list = []
        self.coordinates_list_label.setText("")
        self.mean_std_label.setText("")

        for item in self.plot_widget.items():
            if isinstance(item, pg.ScatterPlotItem):
                self.plot_widget.removeItem(item)

    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.wav)"
        )
        if file_path:
            self.audio_file_label.setText(f"Fichier audio chargé : {file_path}")
            self.get_MFCCS_change(file_path)

    def get_MFCCS_change(
        self,
        file_path,
        channelN=0,
        sigSr=10000,
        tStep=0.005,
        winLen=0.025,
        n_mfcc=13,
        n_fft=512,
        removeFirst=1,
        filtCutoff=12,
        filtOrd=6,
    ):
        try:
            myAudio, _ = load(file_path, sr=sigSr, mono=False)
            if myAudio.ndim > 1:
                y = myAudio[channelN, :]
            else:
                y = myAudio

            win_length = int(np.rint(winLen * sigSr))
            hop_length = int(np.rint(tStep * sigSr))
            myMfccs = lf.mfcc(
                y=y,
                sr=sigSr,
                n_mfcc=n_mfcc,
                win_length=win_length,
                hop_length=hop_length,
                n_fft=n_fft,
            )
            if removeFirst:
                myMfccs = myMfccs[1:, :]
            cutOffNorm = filtCutoff / ((1 / tStep) / 2)
            b1, a1 = signal.butter(filtOrd, cutOffNorm, "lowpass")
            filtMffcs = signal.filtfilt(b1, a1, myMfccs, axis=1)
            myAbsDiff = np.sqrt(np.gradient(filtMffcs, axis=1) ** 2)
            totChange = np.sum(myAbsDiff, axis=0)
            totChange = signal.filtfilt(b1, a1, totChange)
            self.modulation_mfcc_plot(totChange)

        except Exception as e:
            print(f"Error loading audio file: {e}")

    def modulation_mfcc_plot(self, valeurs):
        fs = 200
        temps = np.arange(0, len(valeurs)) / fs
        self.ax.clear()
        self.ax.plot(temps, valeurs, pen="#FF5733")
        self.ax.setTitle("Modulation MFCC")
        self.ax.showGrid(x=True, y=True)

        self.region = pg.LinearRegionItem()
        self.plot_widget.addItem(self.region)
        self.region.sigRegionChanged.connect(self.region_changed)

    def add_interval(
        self,
        start_time: float,
        default_duration: float,
        tier_label: str,
        plot,
        parent_tier_name: str,
    ) -> None:
        new_tier_interval = Tier(start_time, start_time + default_duration, tier_label)
        #TODO

    def remove_interval(self, interval, plot, parent_tier_name):
        pass
        #TODO

    def save_annotations(self):
        if self.textgrid is None:
            #TODO Display pop up error
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save TextGrid File", "", "TextGrid Files (*.TextGrid)"
        )

        if not filepath:
            #TODO Display pop up error
            return
        
        tgt_textgrid = self.textgrid.to_textgrid()
        tgt.io.write_to_file(tgt_textgrid, filepath, format="long")

    def load_annotations(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open TextGrid File", "", "TextGrid Files (*.TextGrid)"
        )
        if not filepath:
            #TODO Display pop up error
            return


        tgt_textgrid = tgt.io.read_textgrid(filepath)
        self.textgrid = ui_tgt.TextgridTGTConvert().from_textgrid(tgt_textgrid,
                                                               self.plot_widget)

        self.layout.addWidget(self.textgrid, 2, 0)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = AudioAnalyzer()
    ex.show()
    sys.exit(app.exec_())
