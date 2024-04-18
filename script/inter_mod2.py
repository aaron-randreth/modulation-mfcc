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



class AudioAnalyzer(QMainWindow):
    textgrid: ui_tiers.TextGrid | None = None
    manual_peak_removal_enabled = False

    manual_point_addition_enabled = False  # Variable pour suivre l'état de la fonctionnalité
    moove = False 
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
        self.valeurs_mfcc = None
        self.temps_mfcc = None
        self.init_toolbar()
        self.manual_point_addition_enabled = False 
        self.manual_point_addition_enabled = False 
        

        self.coordinates_label = QLabel("Mesures:")
        coordinates_layout.addWidget(self.coordinates_label)
        self.coordinates_list_label = QLabel()
        coordinates_layout.addWidget(self.coordinates_list_label)
        self.mean_std_label = QLabel()
        coordinates_layout.addWidget(self.mean_std_label)
        self.coordinates_list = []
        self.region = pg.LinearRegionItem()
        self.region.setMovable(True)
        self.plot_widget.addItem(self.region)
        self.duration_text = pg.TextItem(anchor=(0.5, 2.5))
        self.plot_widget.addItem(self.duration_text)
        self.analysis_button = QPushButton("Analyse Maximum")
        self.analysis_button.clicked.connect(self.analyse_maximum)
        self.layout.addWidget(self.analysis_button, 7, 0)  
        toggle_manual_peak_removal_action = QAction("Toggle Manual Peak Removal", self)
        
        toggle_manual_peak_removal_action.setCheckable(True)
        toggle_manual_peak_removal_action.setChecked(False)  # Par défaut, désactivé
        toggle_manual_peak_removal_action.triggered.connect(self.toggle_manual_peak_removal)
        self.toolbar.addAction(toggle_manual_peak_removal_action)

        # Ajoutez un bouton pour activer/désactiver l'ajout de points
        toggle_manual_point_addition_action = QAction("Toggle Manual Point Addition", self)
        toggle_manual_point_addition_action.setCheckable(True)
        toggle_manual_point_addition_action.setChecked(False)  # Par défaut, désactivé
        toggle_manual_point_addition_action.triggered.connect(self.toggle_manual_point_addition)
        self.toolbar.addAction(toggle_manual_point_addition_action)
#ici le mouseclick
        self.plot_widget.scene().sigMouseClicked.connect(self.remove_peak_on_click)
        self.plot_widget.scene().sigMouseClicked.connect(self.add_point_on_click)

        def update_duration():
            region_values = self.region.getRegion()
            duration = region_values[1] - region_values[0]
            self.duration_text.setText(f"Durée sélectionnée : {duration:.2f} s")

        self.region.sigRegionChanged.connect(update_duration)
        update_duration()

        self.tiers = {}

    def init_toolbar(self):
        self.toolbar = QToolBar("Tools")
        self.addToolBar(self.toolbar)


        # Action pour déplacer un pic
        move_action = QAction("Move Peak", self)
        move_action.triggered.connect(self.move_peak)
        self.toolbar.addAction(move_action)

    def toggle_manual_peak_removal(self, checked):
        # Méthode pour activer ou désactiver la suppression manuelle des pics
        self.manual_peak_removal_enabled = checked



    def toggle_manual_point_addition(self, checked):
        # Méthode pour activer ou désactiver l'ajout de points
        self.manual_point_addition_enabled = checked

    def add_point_on_click(self, event):
        if self.manual_point_addition_enabled:
            mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(
                event.scenePos()
            )
            x, y = mouse_point.x(), mouse_point.y()

            # Recherche du point sur la courbe dont la coordonnée x est la plus proche de la coordonnée x de la souris
            closest_x_index = np.argmin(np.abs(self.temps_mfcc - x))
            closest_x = self.temps_mfcc[closest_x_index]
            closest_y = self.valeurs_mfcc[closest_x_index]

            # Ajout du point trouvé sur le graphique et dans la liste des coordonnées
            self.coordinates_list.append((closest_x, closest_y))

            # Mise à jour de l'affichage des coordonnées
            coordinates_text = "\n".join(
                [
                    f"X: {coord[0]:.2f}, Y: {coord[1]:.2f}"
                    for coord in self.coordinates_list
                ]
            )
            self.coordinates_list_label.setText(
                f"Last Coordinate: X: {closest_x:.2f}, Y: {closest_y:.2f}\n{coordinates_text}"
            )

            points_x, points_y = self.mfcc_points.getData()

            self.mfcc_points.setData(
                    np.append(points_x, closest_x),
                    np.append(points_y, closest_y)
            )



    def remove_peak_on_click(self, event):
        if not self.manual_peak_removal_enabled:
            return

        self.plot_widget.setCursor(Qt.PointingHandCursor)

        mouse_point = self.ax.vb.mapSceneToView(event.scenePos())
        click_x = mouse_point.x()
        click_y = mouse_point.y()

        distance_threshold = 1 

        points_data = self.mfcc_points.getData()
        closest_distance = float('inf')
        closest_index = None

        for i, (point_x, point_y) in enumerate(zip(*points_data)):
            distance = ((point_x - click_x) ** 2 + (point_y - click_y) ** 2) ** 0.5
            if distance < closest_distance:
                closest_distance = distance
                closest_index = i

        if closest_index is None:
            return

        if closest_distance > distance_threshold:
            return

        updated_x = [x for j, x in enumerate(points_data[0]) if j != closest_index]
        updated_y = [y for j, y in enumerate(points_data[1]) if j != closest_index]

        self.mfcc_points.setData(updated_x, updated_y)


    def move_peak(self):
        # Logique pour déplacer un pic
        pass
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


    def find_maxima_in_interval(interval_values, time, min_distance=None):
        if min_distance is None:
            maxima_indices, _ = find_peaks(interval_values)
            print(interval_values)
            peak_times = time[maxima_indices]
            time_gaps = np.diff(peak_times)
            print(peak_times)
            if len(time_gaps) > 0:
                q75, q10 = np.percentile(time_gaps, [75,5])
                iqr = q75 - q10 -0.03
                min_distance = iqr
                print(iqr)
            else:
                min_distance = 0
        print(interval_values)
        min_distance_points = max(int(min_distance * len(interval_values) / (time[-1] - time[0])), 1)
        
        maxima_indices, _ = find_peaks(interval_values, distance=min_distance_points, prominence=5)
        maxima_times = time[maxima_indices]
        maxima_values = interval_values[maxima_indices]

        if len(maxima_values) > 2:
            avg_max = np.mean(maxima_values[1:-1])
        else:
            avg_max = np.mean(maxima_values)

        return len(maxima_indices[1:-1]), maxima_times, avg_max
    def get_selected_tier_interval(self):
        if self.textgrid is None:
            return

        first_tier = self.textgrid.get_tier_by_index(0)
        for t in first_tier.get_elements():
            if not t.get_name():
                continue

            return t

        return None
        
############# ICI se trouve a la fois le moment ou on prend les valeurs de la modulation , et on effecture un peak picking basique je vais rajouter le percentil etc.
    def analyse_maximum(self):
        if self.textgrid is None or self.valeurs_mfcc is None or self.temps_mfcc is None:
            return

        selected_interval = self.get_selected_tier_interval()
        if selected_interval is None:
            return

        start_time = float(selected_interval.start_time)
        end_time = float(selected_interval.end_time)
        fs = 200  
        start_index = int(start_time * fs)
        end_index = int(end_time * fs)

        start_index = max(start_index, 0)
        end_index = min(end_index, len(self.valeurs_mfcc))

        interval_values = self.valeurs_mfcc[start_index:end_index]
        interval_time = self.temps_mfcc[start_index:end_index]


        min_peaks, _ = find_peaks(-interval_values)

        initial_peaks, _ = find_peaks(interval_values)
        if len(initial_peaks) > 1:
            peak_times = interval_time[initial_peaks]
            time_gaps = np.diff(peak_times)

            q75, q25 = np.percentile(time_gaps, [100 ,10])
            iqr = q75 - q25
            min_distance_time = iqr - 0.03  
            min_distance_samples = max(int(min_distance_time * len(interval_values) / (interval_time[-1] - interval_time[0])), 1)

            peaks, _ = find_peaks(interval_values, distance=min_distance_samples, prominence=1)
            peak_times_final = interval_time[peaks]
            peak_values_final = interval_values[peaks]

            # Vérifier si le dernier maximum dépasse le dernier minimum
            if len(min_peaks) > 0 and len(peak_values_final) > 0:
                last_max_index = peaks[-1]
                last_min_index = min_peaks[-1]
                if last_max_index > last_min_index:
                    print("Le dernier maximum dépasse le dernier minimum.")
                    peak_times_final = np.delete(peak_times_final, -1)
                    peak_values_final = np.delete(peak_values_final, -1)
        else:
            peaks = initial_peaks
            peak_times_final = interval_time[peaks] if len(peaks) > 0 else []
            peak_values_final = interval_values[peaks] if len(peaks) > 0 else []

        print("Filtered Peaks times:", peak_times_final)
        print("Filtered Peaks values:", peak_values_final)
        if len(peak_times_final) > 0 and len(peak_values_final) > 0:

            self.mfcc_points = pg.ScatterPlotItem(
                    x=peak_times_final,
                    y=peak_values_final,
                    symbol="x",
                    size=10,
                    pen=pg.mkPen('g'),  
                    brush=pg.mkBrush('b'),
                )

            self.plot_widget.addItem(self.mfcc_points)


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
            self.valeurs_mfcc = totChange

            temps = np.arange(len(totChange)) / 200.0  
            self.temps_mfcc = temps

            self.modulation_mfcc_plot(totChange)
            print(totChange)
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
        self.valeurs_mfcc = valeurs  # Stockez les valeurs MFCC dans la classe
        self.temps_mfcc = temps

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
