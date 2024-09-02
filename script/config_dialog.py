from PyQt5 import QtWidgets, QtCore, QtGui
import json

class UnifiedConfigDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Parameters")
        self.layout = QtWidgets.QVBoxLayout(self)
        # Définir une police plus petite pour toute la fenêtre de dialogue
        font = QtGui.QFont()
        font.setPointSize(9)
        self.setFont(font)
        # Dropdown menu for selecting the configuration type
        self.config_type_combo = QtWidgets.QComboBox()
        self.config_type_combo.addItems([
            "MFCC Configuration", 
            "Amplitude Configuration", 
            "Formant1 Configuration", 
            "Formant2 Configuration", 
            "Formant3 Configuration", 
            "F0 Configuration",
                    "EMA Configuration" 
        ])
        self.config_type_combo.currentIndexChanged.connect(self.display_selected_config)

        self.config_stack = QtWidgets.QStackedWidget()

        self.mfcc_widget = self.create_mfcc_widget()
        self.config_stack.addWidget(self.mfcc_widget)

        self.amp_widget = self.create_amp_widget()
        self.config_stack.addWidget(self.amp_widget)

        self.formant1_widget = self.create_formant1_widget()
        self.config_stack.addWidget(self.formant1_widget)

        self.formant2_widget = self.create_formant2_widget()
        self.config_stack.addWidget(self.formant2_widget)

        self.formant3_widget = self.create_formant3_widget()
        self.config_stack.addWidget(self.formant3_widget)

        self.f0_widget = self.create_f0_widget()
        self.config_stack.addWidget(self.f0_widget)
        self.ema_widget = self.create_ema_widget()  
        self.config_stack.addWidget(self.ema_widget)

        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.clicked.connect(self.accept)
        self.save_button = QtWidgets.QPushButton("Save Config")
        self.save_button.clicked.connect(self.save_config)
        self.load_button = QtWidgets.QPushButton("Load Config")
        self.load_button.clicked.connect(self.load_config)

        self.layout.addWidget(self.config_type_combo)
        self.layout.addWidget(self.config_stack)
        self.layout.addWidget(self.apply_button)
        self.layout.addWidget(self.save_button)
        self.layout.addWidget(self.load_button)

        self.setLayout(self.layout)

    def display_selected_config(self, index):
        """Displays the selected configuration panel and ensures that all fields are enabled/disabled appropriately."""
        self.config_stack.setCurrentIndex(index)

        if index == 0:  # MFCC
            self.toggle_mfcc_fields(self.mfcc_enable_checkbox.checkState())
        elif index == 1:  # Amplitude
            self.toggle_amp_fields(self.amp_enable_checkbox.checkState())
        elif index == 2:  # Formant1
            self.toggle_formant1_fields(self.formant1_enable_checkbox.checkState())
        elif index == 3:  # Formant2
            self.toggle_formant2_fields(self.formant2_enable_checkbox.checkState())
        elif index == 4:  # Formant3
            self.toggle_formant3_fields(self.formant3_enable_checkbox.checkState())
        elif index == 5:  # F0
            self.toggle_f0_fields(self.f0_enable_checkbox.checkState())
        elif index == 6:  # EMA

            self.ema_target_sample_rate_input[1].setEnabled(True)
    def create_ema_widget(self):
        """Create the EMA configuration widget."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(widget)

        # EMA controls
        self.ema_target_sample_rate_input = self.create_input_field("Target Sample Rate (Hz):", "200")

        # EMA Derivative Controls
        self.ema_derivative_method_input = self.create_input_field("Derivative Method (grad/sg/finDiff):", "gradient")
        self.ema_width_input = self.create_input_field("SG Width:", "3")
        self.ema_acc_order_input = self.create_input_field("Finite Difference Accuracy Order:", "2")
        self.ema_poly_order_input = self.create_input_field("SG Polynomial Order:", "2")

        # Add EMA widgets to layout
        self.add_groupbox_to_layout("EMA Configuration", [
            self.ema_target_sample_rate_input,
            self.ema_derivative_method_input,
            self.ema_width_input,
            self.ema_acc_order_input,
            self.ema_poly_order_input,
        ], layout, 0, 0)

        return widget


    def create_mfcc_widget(self):
        """Create the MFCC configuration widget."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(widget)

        # MFCC controls
        self.mfcc_enable_checkbox = QtWidgets.QCheckBox("Enable Mod Cepstr Customization")
        self.mfcc_enable_checkbox.setChecked(False)
        self.mfcc_enable_checkbox.stateChanged.connect(self.toggle_mfcc_fields)

        self.mfcc_sample_rate_input = self.create_input_field("Sample Rate (Hz):", "10000")
        self.mfcc_tstep_input = self.create_input_field("Time Step (s):", "0.005")
        self.mfcc_winlen_input = self.create_input_field("Window Length (s):", "0.025")
        self.mfcc_nmfcc_input = self.create_input_field("Number of MFCCs:", "13")
        self.mfcc_nfft_input = self.create_input_field("Number of FFT Points:", "512")
        self.mfcc_remove_first_input = self.create_input_field("Remove First MFCC (1/0):", "1")
        self.mfcc_filt_cutoff_input = self.create_input_field("Filter Cutoff Frequency (Hz):", "12")
        self.mfcc_filt_ord_input = self.create_input_field("Filter Order:", "6")
        self.mfcc_diff_method_input = self.create_input_field("Diff Method (grad/sg):", "grad")
        self.mfcc_out_filter_input = self.create_input_field("Output Filter (None/iir/fir/sg):", "iir")
        self.mfcc_out_filt_type_input = self.create_input_field("Filter Type (low/band):", "low")
        self.mfcc_out_filt_cutoff_input = self.create_input_field("Output Filter Cutoff (Hz):", "12")
        self.mfcc_out_filt_len_input = self.create_input_field("Filter Length:", "6")
        self.mfcc_out_filt_polyord_input = self.create_input_field("Filter Polynomial Order:", "3")
        self.mfcc_name_input = self.create_input_field("Curve Name:", "Custom MFCC")
        self.mfcc_panel_choice = QtWidgets.QComboBox()
        self.mfcc_panel_choice.addItems(["1", "2", "3", "4"])

        self.mfcc_derivative_group = QtWidgets.QButtonGroup()
        self.mfcc_traj_radio = QtWidgets.QRadioButton("Traj")
        self.mfcc_vel_radio = QtWidgets.QRadioButton("Vel")
        self.mfcc_acc_radio = QtWidgets.QRadioButton("Acc")
        self.mfcc_traj_radio.setChecked(True)
        self.mfcc_derivative_group.addButton(self.mfcc_traj_radio)
        self.mfcc_derivative_group.addButton(self.mfcc_vel_radio)
        self.mfcc_derivative_group.addButton(self.mfcc_acc_radio)
        self.mfcc_derivative_group.buttonClicked.connect(self.toggle_mfcc_derivative_fields)

        derivative_layout = QtWidgets.QVBoxLayout()
        derivative_layout.addWidget(self.mfcc_traj_radio)
        derivative_layout.addWidget(self.mfcc_vel_radio)
        derivative_layout.addWidget(self.mfcc_acc_radio)
        self.mfcc_derivative_widget = QtWidgets.QWidget()
        self.mfcc_derivative_widget.setLayout(derivative_layout)

        self.mfcc_derivative_method_input = self.create_input_field("Derivative Method (grad/sg/finDiff):", "gradient")
        self.mfcc_width_input = self.create_input_field("SG Width:", "3")
        self.mfcc_acc_order_input = self.create_input_field("Finite Difference Accuracy Order:", "2")
        self.mfcc_poly_order_input = self.create_input_field("SG Polynomial Order:", "2")

        self.toggle_mfcc_fields(self.mfcc_enable_checkbox.checkState())

        self.add_groupbox_to_layout("MFCC Configuration", [
            self.mfcc_enable_checkbox,
            self.mfcc_sample_rate_input,
            self.mfcc_tstep_input,
            self.mfcc_winlen_input,
            self.mfcc_nmfcc_input,
            self.mfcc_nfft_input,
            self.mfcc_remove_first_input,
            self.mfcc_filt_cutoff_input,
            self.mfcc_filt_ord_input,
            self.mfcc_diff_method_input,
            self.mfcc_out_filter_input,
            self.mfcc_out_filt_type_input,
            self.mfcc_out_filt_cutoff_input,
            self.mfcc_out_filt_len_input,
            self.mfcc_out_filt_polyord_input,
            self.mfcc_name_input,
            (QtWidgets.QLabel("MFCC Panel:"), self.mfcc_panel_choice),
            self.mfcc_derivative_widget,
            self.mfcc_derivative_method_input,
            self.mfcc_width_input,
            self.mfcc_acc_order_input,
            self.mfcc_poly_order_input,
        ], layout, 0, 0)

        return widget

    def create_amp_widget(self):
        """Create the Amplitude configuration widget."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(widget)

        # Amplitude controls
        self.amp_enable_checkbox = QtWidgets.QCheckBox("Enable ENV AMP Customization")
        self.amp_enable_checkbox.setChecked(False)
        self.amp_enable_checkbox.stateChanged.connect(self.toggle_amp_fields)

        self.amp_method_input = self.create_input_field("Method (RMS/Hilb/RMSpraat):", "RMS")
        self.amp_winlen_input = self.create_input_field("Window Length (s):", "0.1")
        self.amp_hoplen_input = self.create_input_field("Hop Length (s):", "0.005")
        self.amp_center_input = self.create_input_field("Center (True/False):", "True")
        self.amp_outfilter_input = self.create_input_field("Output Filter (None/iir/fir/sg):", "None")
        self.amp_outfilt_type_input = self.create_input_field("Filter Type (low/band):", "low")
        self.amp_outfilt_cutoff_input = self.create_input_field("Filter Cutoff Frequency (Hz):", "12")
        self.amp_outfilt_len_input = self.create_input_field("Filter Length:", "6")
        self.amp_outfilt_polyord_input = self.create_input_field("Filter Polynomial Order:", "3")
        self.amp_name_input = self.create_input_field("Curve Name:", "Custom Amplitude")
        self.amp_panel_choice = QtWidgets.QComboBox()
        self.amp_panel_choice.addItems(["1", "2", "3", "4"])

        self.amp_derivative_group = QtWidgets.QButtonGroup()
        self.amp_traj_radio = QtWidgets.QRadioButton("Traj")
        self.amp_vel_radio = QtWidgets.QRadioButton("Vel")
        self.amp_acc_radio = QtWidgets.QRadioButton("Acc")
        self.amp_traj_radio.setChecked(True)
        self.amp_derivative_group.addButton(self.amp_traj_radio)
        self.amp_derivative_group.addButton(self.amp_vel_radio)
        self.amp_derivative_group.addButton(self.amp_acc_radio)
        self.amp_derivative_group.buttonClicked.connect(self.toggle_amp_derivative_fields)

        derivative_layout = QtWidgets.QVBoxLayout()
        derivative_layout.addWidget(self.amp_traj_radio)
        derivative_layout.addWidget(self.amp_vel_radio)
        derivative_layout.addWidget(self.amp_acc_radio)
        self.amp_derivative_widget = QtWidgets.QWidget()
        self.amp_derivative_widget.setLayout(derivative_layout)

        self.amp_derivative_method_input = self.create_input_field("Derivative Method (grad/sg/finDiff):", "gradient")
        self.amp_width_input = self.create_input_field("SG Width:", "3")
        self.amp_acc_order_input = self.create_input_field("Finite Difference Accuracy Order:", "2")
        self.amp_poly_order_input = self.create_input_field("SG Polynomial Order:", "2")

        self.toggle_amp_fields(self.amp_enable_checkbox.checkState())

        self.add_groupbox_to_layout("Amplitude Configuration", [
            self.amp_enable_checkbox,
            self.amp_method_input,
            self.amp_winlen_input,
            self.amp_hoplen_input,
            self.amp_center_input,
            self.amp_outfilter_input,
            self.amp_outfilt_type_input,
            self.amp_outfilt_cutoff_input,
            self.amp_outfilt_len_input,
            self.amp_outfilt_polyord_input,
            self.amp_name_input,
            (QtWidgets.QLabel("Amplitude Panel:"), self.amp_panel_choice),
            self.amp_derivative_widget,
            self.amp_derivative_method_input,
            self.amp_width_input,
            self.amp_acc_order_input,
            self.amp_poly_order_input,
        ], layout, 0, 0)

        return widget

    def create_formant1_widget(self):
        """Create the Formant1 configuration widget."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(widget)

        # Formant1 controls
        self.formant1_enable_checkbox = QtWidgets.QCheckBox("Enable Formant1 Customization")
        self.formant1_enable_checkbox.setChecked(False)
        self.formant1_enable_checkbox.stateChanged.connect(self.toggle_formant1_fields)

        self.formant1_energy_threshold_input = self.create_input_field("Energy Threshold:", "40.0")
        self.formant1_tstep_input = self.create_input_field("Time Step (s):", "0.005")
        self.formant1_max_num_formants_input = self.create_input_field("Max Number of Formants:", "5")
        self.formant1_max_formant_input = self.create_input_field("Maximum Formant (Hz):", "5500.0")
        self.formant1_winlen_input = self.create_input_field("Window Length (s):", "0.025")
        self.formant1_pre_emphasis_input = self.create_input_field("Pre-emphasis From (Hz):", "50.0")
        self.formant1_name_input = self.create_input_field("Curve Name:", "Custom Formant1")
        self.formant1_panel_choice = QtWidgets.QComboBox()
        self.formant1_panel_choice.addItems(["1", "2", "3", "4"])

        self.formant1_derivative_group = QtWidgets.QButtonGroup()
        self.formant1_traj_radio = QtWidgets.QRadioButton("Traj")
        self.formant1_vel_radio = QtWidgets.QRadioButton("Vel")
        self.formant1_acc_radio = QtWidgets.QRadioButton("Acc")
        self.formant1_traj_radio.setChecked(True)
        self.formant1_derivative_group.addButton(self.formant1_traj_radio)
        self.formant1_derivative_group.addButton(self.formant1_vel_radio)
        self.formant1_derivative_group.addButton(self.formant1_acc_radio)
        self.formant1_derivative_group.buttonClicked.connect(self.toggle_formant1_derivative_fields)

        derivative_layout = QtWidgets.QVBoxLayout()
        derivative_layout.addWidget(self.formant1_traj_radio)
        derivative_layout.addWidget(self.formant1_vel_radio)
        derivative_layout.addWidget(self.formant1_acc_radio)
        self.formant1_derivative_widget = QtWidgets.QWidget()
        self.formant1_derivative_widget.setLayout(derivative_layout)

        self.formant1_derivative_method_input = self.create_input_field("Derivative Method (grad/sg/finDiff):", "gradient")
        self.formant1_width_input = self.create_input_field("SG Width:", "3")
        self.formant1_acc_order_input = self.create_input_field("Finite Difference Accuracy Order:", "2")
        self.formant1_poly_order_input = self.create_input_field("SG Polynomial Order:", "2")

        self.toggle_formant1_fields(self.formant1_enable_checkbox.checkState())

        self.add_groupbox_to_layout("Formant1 Configuration", [
            self.formant1_enable_checkbox,
            self.formant1_energy_threshold_input,
            self.formant1_tstep_input,
            self.formant1_max_num_formants_input,
            self.formant1_max_formant_input,
            self.formant1_winlen_input,
            self.formant1_pre_emphasis_input,
            self.formant1_name_input,
            (QtWidgets.QLabel("Formant1 Panel:"), self.formant1_panel_choice),
            self.formant1_derivative_widget,
            self.formant1_derivative_method_input,
            self.formant1_width_input,
            self.formant1_acc_order_input,
            self.formant1_poly_order_input,
        ], layout, 0, 0)

        return widget

    def create_formant2_widget(self):
        """Create the Formant2 configuration widget."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(widget)

        # Formant2 controls
        self.formant2_enable_checkbox = QtWidgets.QCheckBox("Enable Formant2 Customization")
        self.formant2_enable_checkbox.setChecked(False)
        self.formant2_enable_checkbox.stateChanged.connect(self.toggle_formant2_fields)

        self.formant2_energy_threshold_input = self.create_input_field("Energy Threshold:", "40.0")
        self.formant2_tstep_input = self.create_input_field("Time Step (s):", "0.005")
        self.formant2_max_num_formants_input = self.create_input_field("Max Number of Formants:", "5")
        self.formant2_max_formant_input = self.create_input_field("Maximum Formant (Hz):", "5500.0")
        self.formant2_winlen_input = self.create_input_field("Window Length (s):", "0.025")
        self.formant2_pre_emphasis_input = self.create_input_field("Pre-emphasis From (Hz):", "50.0")
        self.formant2_name_input = self.create_input_field("Curve Name:", "Custom Formant2")
        self.formant2_panel_choice = QtWidgets.QComboBox()
        self.formant2_panel_choice.addItems(["1", "2", "3", "4"])

        self.formant2_derivative_group = QtWidgets.QButtonGroup()
        self.formant2_traj_radio = QtWidgets.QRadioButton("Traj")
        self.formant2_vel_radio = QtWidgets.QRadioButton("Vel")
        self.formant2_acc_radio = QtWidgets.QRadioButton("Acc")
        self.formant2_traj_radio.setChecked(True)
        self.formant2_derivative_group.addButton(self.formant2_traj_radio)
        self.formant2_derivative_group.addButton(self.formant2_vel_radio)
        self.formant2_derivative_group.addButton(self.formant2_acc_radio)
        self.formant2_derivative_group.buttonClicked.connect(self.toggle_formant2_derivative_fields)

        derivative_layout = QtWidgets.QVBoxLayout()
        derivative_layout.addWidget(self.formant2_traj_radio)
        derivative_layout.addWidget(self.formant2_vel_radio)
        derivative_layout.addWidget(self.formant2_acc_radio)
        self.formant2_derivative_widget = QtWidgets.QWidget()
        self.formant2_derivative_widget.setLayout(derivative_layout)

        self.formant2_derivative_method_input = self.create_input_field("Derivative Method (grad/sg/finDiff):", "gradient")
        self.formant2_width_input = self.create_input_field("SG Width:", "3")
        self.formant2_acc_order_input = self.create_input_field("Finite Difference Accuracy Order:", "2")
        self.formant2_poly_order_input = self.create_input_field("SG Polynomial Order:", "2")

        self.toggle_formant2_fields(self.formant2_enable_checkbox.checkState())

        self.add_groupbox_to_layout("Formant2 Configuration", [
            self.formant2_enable_checkbox,
            self.formant2_energy_threshold_input,
            self.formant2_tstep_input,
            self.formant2_max_num_formants_input,
            self.formant2_max_formant_input,
            self.formant2_winlen_input,
            self.formant2_pre_emphasis_input,
            self.formant2_name_input,
            (QtWidgets.QLabel("Formant2 Panel:"), self.formant2_panel_choice),
            self.formant2_derivative_widget,
            self.formant2_derivative_method_input,
            self.formant2_width_input,
            self.formant2_acc_order_input,
            self.formant2_poly_order_input,
        ], layout, 0, 0)

        return widget

    def create_formant3_widget(self):
        """Create the Formant3 configuration widget."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(widget)

        # Formant3 controls
        self.formant3_enable_checkbox = QtWidgets.QCheckBox("Enable Formant3 Customization")
        self.formant3_enable_checkbox.setChecked(False)
        self.formant3_enable_checkbox.stateChanged.connect(self.toggle_formant3_fields)

        self.formant3_energy_threshold_input = self.create_input_field("Energy Threshold:", "40.0")
        self.formant3_tstep_input = self.create_input_field("Time Step (s):", "0.005")
        self.formant3_max_num_formants_input = self.create_input_field("Max Number of Formants:", "5")
        self.formant3_max_formant_input = self.create_input_field("Maximum Formant (Hz):", "5500.0")
        self.formant3_winlen_input = self.create_input_field("Window Length (s):", "0.025")
        self.formant3_pre_emphasis_input = self.create_input_field("Pre-emphasis From (Hz):", "50.0")
        self.formant3_name_input = self.create_input_field("Curve Name:", "Custom Formant3")
        self.formant3_panel_choice = QtWidgets.QComboBox()
        self.formant3_panel_choice.addItems(["1", "2", "3", "4"])

        self.formant3_derivative_group = QtWidgets.QButtonGroup()
        self.formant3_traj_radio = QtWidgets.QRadioButton("Traj")
        self.formant3_vel_radio = QtWidgets.QRadioButton("Vel")
        self.formant3_acc_radio = QtWidgets.QRadioButton("Acc")
        self.formant3_traj_radio.setChecked(True)
        self.formant3_derivative_group.addButton(self.formant3_traj_radio)
        self.formant3_derivative_group.addButton(self.formant3_vel_radio)
        self.formant3_derivative_group.addButton(self.formant3_acc_radio)
        self.formant3_derivative_group.buttonClicked.connect(self.toggle_formant3_derivative_fields)

        derivative_layout = QtWidgets.QVBoxLayout()
        derivative_layout.addWidget(self.formant3_traj_radio)
        derivative_layout.addWidget(self.formant3_vel_radio)
        derivative_layout.addWidget(self.formant3_acc_radio)
        self.formant3_derivative_widget = QtWidgets.QWidget()
        self.formant3_derivative_widget.setLayout(derivative_layout)

        self.formant3_derivative_method_input = self.create_input_field("Derivative Method (grad/sg/finDiff):", "gradient")
        self.formant3_width_input = self.create_input_field("SG Width:", "3")
        self.formant3_acc_order_input = self.create_input_field("Finite Difference Accuracy Order:", "2")
        self.formant3_poly_order_input = self.create_input_field("SG Polynomial Order:", "2")

        self.toggle_formant3_fields(self.formant3_enable_checkbox.checkState())

        self.add_groupbox_to_layout("Formant3 Configuration", [
            self.formant3_enable_checkbox,
            self.formant3_energy_threshold_input,
            self.formant3_tstep_input,
            self.formant3_max_num_formants_input,
            self.formant3_max_formant_input,
            self.formant3_winlen_input,
            self.formant3_pre_emphasis_input,
            self.formant3_name_input,
            (QtWidgets.QLabel("Formant3 Panel:"), self.formant3_panel_choice),
            self.formant3_derivative_widget,
            self.formant3_derivative_method_input,
            self.formant3_width_input,
            self.formant3_acc_order_input,
            self.formant3_poly_order_input,
        ], layout, 0, 0)

        return widget

    def create_f0_widget(self):
        """Create the F0 configuration widget."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(widget)

        # F0 controls
        self.f0_enable_checkbox = QtWidgets.QCheckBox("Enable F0 Customization")
        self.f0_enable_checkbox.setChecked(False)
        self.f0_enable_checkbox.stateChanged.connect(self.toggle_f0_fields)

        self.f0_method_input = self.create_input_field("Method (praatac/pyin):", "praatac")
        self.f0_hop_size_input = self.create_input_field("Hop Size (s):", "0.005")
        self.f0_min_pitch_input = self.create_input_field("Min Pitch (Hz):", "75")
        self.f0_max_pitch_input = self.create_input_field("Max Pitch (Hz):", "600")
        self.f0_interp_unvoiced_input = self.create_input_field("Interp Unvoiced (None/linear):", "linear")
        self.f0_out_filter_input = self.create_input_field("Output Filter (None/iir/fir/sg):", "iir")
        self.f0_out_filt_type_input = self.create_input_field("Filter Type (low/band/high):", "low")
        self.f0_out_filt_cutoff_input = self.create_input_field("Output Filter Cutoff (Hz):", "12")
        self.f0_out_filt_len_input = self.create_input_field("Filter Length:", "6")
        self.f0_out_filt_polyord_input = self.create_input_field("Filter Polynomial Order:", "3")
        self.f0_name_input = self.create_input_field("Curve Name:", "Custom F0")
        self.f0_panel_choice = QtWidgets.QComboBox()
        self.f0_panel_choice.addItems(["1", "2", "3", "4"])

        self.f0_derivative_group = QtWidgets.QButtonGroup()
        self.f0_traj_radio = QtWidgets.QRadioButton("Traj")
        self.f0_vel_radio = QtWidgets.QRadioButton("Vel")
        self.f0_acc_radio = QtWidgets.QRadioButton("Acc")
        self.f0_traj_radio.setChecked(True)
        self.f0_derivative_group.addButton(self.f0_traj_radio)
        self.f0_derivative_group.addButton(self.f0_vel_radio)
        self.f0_derivative_group.addButton(self.f0_acc_radio)
        self.f0_derivative_group.buttonClicked.connect(self.toggle_f0_derivative_fields)

        derivative_layout = QtWidgets.QVBoxLayout()
        derivative_layout.addWidget(self.f0_traj_radio)
        derivative_layout.addWidget(self.f0_vel_radio)
        derivative_layout.addWidget(self.f0_acc_radio)
        self.f0_derivative_widget = QtWidgets.QWidget()
        self.f0_derivative_widget.setLayout(derivative_layout)

        self.f0_derivative_method_input = self.create_input_field("Derivative Method (grad/sg/finDiff):", "gradient")
        self.f0_width_input = self.create_input_field("SG Width:", "3")
        self.f0_acc_order_input = self.create_input_field("Finite Difference Accuracy Order:", "2")
        self.f0_poly_order_input = self.create_input_field("SG Polynomial Order:", "2")

        self.toggle_f0_fields(self.f0_enable_checkbox.checkState())

        self.add_groupbox_to_layout("F0 Configuration", [
            self.f0_enable_checkbox,
            self.f0_method_input,
            self.f0_hop_size_input,
            self.f0_min_pitch_input,
            self.f0_max_pitch_input,
            self.f0_interp_unvoiced_input,
            self.f0_out_filter_input,
            self.f0_out_filt_type_input,
            self.f0_out_filt_cutoff_input,
            self.f0_out_filt_len_input,
            self.f0_out_filt_polyord_input,
            self.f0_name_input,
            (QtWidgets.QLabel("F0 Panel:"), self.f0_panel_choice),
            self.f0_derivative_widget,
            self.f0_derivative_method_input,
            self.f0_width_input,
            self.f0_acc_order_input,
            self.f0_poly_order_input,
        ], layout, 0, 0)

        return widget

    def create_input_field(self, label_text, default_value):
        label = QtWidgets.QLabel(label_text)
        input_field = QtWidgets.QLineEdit(default_value)
        container = QtWidgets.QVBoxLayout()
        container.addWidget(label)
        container.addWidget(input_field)
        container.setSpacing(1)
        container.setContentsMargins(1, 1, 1, 1)
        widget = QtWidgets.QWidget()
        widget.setLayout(container)
        return widget, input_field

    def add_groupbox_to_layout(self, title, widgets, layout, row, col):
        group_box = QtWidgets.QGroupBox(title)
        group_box_layout = QtWidgets.QVBoxLayout()
        group_box_layout.setSpacing(1)
        group_box_layout.setContentsMargins(1, 1, 1, 1)
        group_box.setLayout(group_box_layout)

        for widget in widgets:
            if isinstance(widget, tuple):
                h_layout = QtWidgets.QHBoxLayout()
                h_layout.addWidget(widget[0])
                h_layout.addWidget(widget[1])
                h_layout.setSpacing(1)
                container = QtWidgets.QWidget()
                container.setLayout(h_layout)
                group_box_layout.addWidget(container)
            else:
                group_box_layout.addWidget(widget)

        layout.addWidget(group_box, row, col)

    def save_config(self):
        params = self.get_parameters()
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Config", "", "JSON Files (*.json);;All Files (*)", options=options)
        if file_name:
            with open(file_name, 'w') as file:
                json.dump(params, file, indent=4)

    def load_config(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Config", "", "JSON Files (*.json);;All Files (*)", options=options)
        if file_name:
            with open(file_name, 'r') as file:
                params = json.load(file)
                self.set_parameters(params)

    def get_parameters(self):
        """Returns the parameters from all configurations."""
        mfcc_enabled = self.mfcc_enable_checkbox.isChecked()
        amp_enabled = self.amp_enable_checkbox.isChecked()
        formant1_enabled = self.formant1_enable_checkbox.isChecked()
        formant2_enabled = self.formant2_enable_checkbox.isChecked()
        formant3_enabled = self.formant3_enable_checkbox.isChecked()
        f0_enabled = self.f0_enable_checkbox.isChecked()

        ema_target_sample_rate = int(self.ema_target_sample_rate_input[1].text())
        print("Saving EMA target sample rate:", ema_target_sample_rate)  # Debug

        params = {
            "mfcc": {
                "enabled": mfcc_enabled,
                "signal_sample_rate": int(self.mfcc_sample_rate_input[1].text()),
                "tStep": float(self.mfcc_tstep_input[1].text()),
                "winLen": float(self.mfcc_winlen_input[1].text()),
                "n_mfcc": int(self.mfcc_nmfcc_input[1].text()),
                "n_fft": int(self.mfcc_nfft_input[1].text()),
                "removeFirst": int(self.mfcc_remove_first_input[1].text()),
                "filtCutoff": float(self.mfcc_filt_cutoff_input[1].text()),
                "filtOrd": int(self.mfcc_filt_ord_input[1].text()),
                "diffMethod": self.mfcc_diff_method_input[1].text(),
                "outFilter": None if self.mfcc_out_filter_input[1].text().lower() == 'none' else self.mfcc_out_filter_input[1].text(),
                "outFiltType": self.mfcc_out_filt_type_input[1].text(),
                "outFiltCutOff": [float(c) for c in self.mfcc_out_filt_cutoff_input[1].text().split()],
                "outFiltLen": int(self.mfcc_out_filt_len_input[1].text()),
                "outFiltPolyOrd": int(self.mfcc_out_filt_polyord_input[1].text()),
                "name": self.mfcc_name_input[1].text(),
                "panel": int(self.mfcc_panel_choice.currentIndex()),
                "derivation_type": 0 if self.mfcc_traj_radio.isChecked() else 1 if self.mfcc_vel_radio.isChecked() else 2,
                "derivative_method": self.mfcc_derivative_method_input[1].text(),
                "sg_width": int(self.mfcc_width_input[1].text()),
                "fin_diff_acc_order": int(self.mfcc_acc_order_input[1].text()),
                "sg_poly_order": int(self.mfcc_poly_order_input[1].text()),
            },
            "amplitude": {
                "enabled": amp_enabled,
                "method": self.amp_method_input[1].text(),
                "winLen": float(self.amp_winlen_input[1].text()),
                "hopLen": float(self.amp_hoplen_input[1].text()),
                "center": self.amp_center_input[1].text().lower() == 'true',
                "outFilter": None if self.amp_outfilter_input[1].text().lower() == 'none' else self.amp_outfilter_input[1].text(),
                "outFiltType": self.amp_outfilt_type_input[1].text(),
                "outFiltCutOff": [float(c) for c in self.amp_outfilt_cutoff_input[1].text().split()],
                "outFiltLen": int(self.amp_outfilt_len_input[1].text()),
                "outFiltPolyOrd": int(self.amp_outfilt_polyord_input[1].text()),
                "name": self.amp_name_input[1].text(),
                "panel": int(self.amp_panel_choice.currentIndex()),
                "derivation_type": 0 if self.amp_traj_radio.isChecked() else 1 if self.amp_vel_radio.isChecked() else 2,
                "derivative_method": self.amp_derivative_method_input[1].text(),
                "sg_width": int(self.amp_width_input[1].text()),
                "fin_diff_acc_order": int(self.amp_acc_order_input[1].text()),
                "sg_poly_order": int(self.amp_poly_order_input[1].text()),
            },
            "formant1": {
                "enabled": formant1_enabled,
                "energy_threshold": float(self.formant1_energy_threshold_input[1].text()),
                "time_step": float(self.formant1_tstep_input[1].text()),
                "max_num_formants": int(self.formant1_max_num_formants_input[1].text()),
                "max_formant": float(self.formant1_max_formant_input[1].text()),
                "window_length": float(self.formant1_winlen_input[1].text()),
                "pre_emphasis_from": float(self.formant1_pre_emphasis_input[1].text()),
                "name": self.formant1_name_input[1].text(),
                "panel": int(self.formant1_panel_choice.currentIndex()),
                "derivation_type": 0 if self.formant1_traj_radio.isChecked() else 1 if self.formant1_vel_radio.isChecked() else 2,
                "derivative_method": self.formant1_derivative_method_input[1].text(),
                "sg_width": int(self.formant1_width_input[1].text()),
                "fin_diff_acc_order": int(self.formant1_acc_order_input[1].text()),
                "sg_poly_order": int(self.formant1_poly_order_input[1].text()),
            },
            "formant2": {
                "enabled": formant2_enabled,
                "energy_threshold": float(self.formant2_energy_threshold_input[1].text()),
                "time_step": float(self.formant2_tstep_input[1].text()),
                "max_num_formants": int(self.formant2_max_num_formants_input[1].text()),
                "max_formant": float(self.formant2_max_formant_input[1].text()),
                "window_length": float(self.formant2_winlen_input[1].text()),
                "pre_emphasis_from": float(self.formant2_pre_emphasis_input[1].text()),
                "name": self.formant2_name_input[1].text(),
                "panel": int(self.formant2_panel_choice.currentIndex()),
                "derivation_type": 0 if self.formant2_traj_radio.isChecked() else 1 if self.formant2_vel_radio.isChecked() else 2,
                "derivative_method": self.formant2_derivative_method_input[1].text(),
                "sg_width": int(self.formant2_width_input[1].text()),
                "fin_diff_acc_order": int(self.formant2_acc_order_input[1].text()),
                "sg_poly_order": int(self.formant2_poly_order_input[1].text()),
            },
            "formant3": {
                "enabled": formant3_enabled,
                "energy_threshold": float(self.formant3_energy_threshold_input[1].text()),
                "time_step": float(self.formant3_tstep_input[1].text()),
                "max_num_formants": int(self.formant3_max_num_formants_input[1].text()),
                "max_formant": float(self.formant3_max_formant_input[1].text()),
                "window_length": float(self.formant3_winlen_input[1].text()),
                "pre_emphasis_from": float(self.formant3_pre_emphasis_input[1].text()),
                "name": self.formant3_name_input[1].text(),
                "panel": int(self.formant3_panel_choice.currentIndex()),
                "derivation_type": 0 if self.formant3_traj_radio.isChecked() else 1 if self.formant3_vel_radio.isChecked() else 2,
                "derivative_method": self.formant3_derivative_method_input[1].text(),
                "sg_width": int(self.formant3_width_input[1].text()),
                "fin_diff_acc_order": int(self.formant3_acc_order_input[1].text()),
                "sg_poly_order": int(self.formant3_poly_order_input[1].text()),
            },
            "f0": {
                "enabled": f0_enabled,
                "method": self.f0_method_input[1].text(),
                "hopSize": float(self.f0_hop_size_input[1].text()),
                "minPitch": float(self.f0_min_pitch_input[1].text()),
                "maxPitch": float(self.f0_max_pitch_input[1].text()),
                "interpUnvoiced": self.f0_interp_unvoiced_input[1].text(),
                "outFilter": None if self.f0_out_filter_input[1].text().lower() == 'none' else self.f0_out_filter_input[1].text(),
                "outFiltType": self.f0_out_filt_type_input[1].text(),
                "outFiltCutOff": [float(c) for c in self.f0_out_filt_cutoff_input[1].text().split()],
                "outFiltLen": int(self.f0_out_filt_len_input[1].text()),
                "outFiltPolyOrd": int(self.f0_out_filt_polyord_input[1].text()),
                "name": self.f0_name_input[1].text(),
                "panel": int(self.f0_panel_choice.currentIndex()),
                "derivation_type": 0 if self.f0_traj_radio.isChecked() else 1 if self.f0_vel_radio.isChecked() else 2,
                "derivative_method": self.f0_derivative_method_input[1].text(),
                "sg_width": int(self.f0_width_input[1].text()),
                "fin_diff_acc_order": int(self.f0_acc_order_input[1].text()),
                "sg_poly_order": int(self.f0_poly_order_input[1].text()),
            },
            "ema": {
                "target_sample_rate": ema_target_sample_rate,
                "derivative_method": self.ema_derivative_method_input[1].text(),
                "sg_width": int(self.ema_width_input[1].text()),
                "fin_diff_acc_order": int(self.ema_acc_order_input[1].text()),
                "sg_poly_order": int(self.ema_poly_order_input[1].text()),
            },

        }
        return params

    def set_parameters(self, params):
        # Setting the MFCC parameters
        mfcc_params = params.get("mfcc", {})
        if mfcc_params.get("enabled"):
            self.mfcc_enable_checkbox.setChecked(True)
            self.mfcc_sample_rate_input[1].setText(str(mfcc_params.get("signal_sample_rate", "")))
            self.mfcc_tstep_input[1].setText(str(mfcc_params.get("tStep", "")))
            self.mfcc_winlen_input[1].setText(str(mfcc_params.get("winLen", "")))
            self.mfcc_nmfcc_input[1].setText(str(mfcc_params.get("n_mfcc", "")))
            self.mfcc_nfft_input[1].setText(str(mfcc_params.get("n_fft", "")))
            self.mfcc_remove_first_input[1].setText(str(mfcc_params.get("removeFirst", "")))
            self.mfcc_filt_cutoff_input[1].setText(str(mfcc_params.get("filtCutoff", "")))
            self.mfcc_filt_ord_input[1].setText(str(mfcc_params.get("filtOrd", "")))
            self.mfcc_diff_method_input[1].setText(mfcc_params.get("diffMethod", ""))
            self.mfcc_out_filter_input[1].setText(mfcc_params.get("outFilter", "None"))
            self.mfcc_out_filt_type_input[1].setText(mfcc_params.get("outFiltType", ""))
            self.mfcc_out_filt_cutoff_input[1].setText(" ".join(map(str, mfcc_params.get("outFiltCutOff", []))))
            self.mfcc_out_filt_len_input[1].setText(str(mfcc_params.get("outFiltLen", "")))
            self.mfcc_out_filt_polyord_input[1].setText(str(mfcc_params.get("outFiltPolyOrd", "")))
            self.mfcc_name_input[1].setText(mfcc_params.get("name", ""))
            self.mfcc_panel_choice.setCurrentIndex(mfcc_params.get("panel", 0))
            self.mfcc_traj_radio.setChecked(mfcc_params.get("derivation_type") == 0)
            self.mfcc_vel_radio.setChecked(mfcc_params.get("derivation_type") == 1)
            self.mfcc_acc_radio.setChecked(mfcc_params.get("derivation_type") == 2)
            self.mfcc_derivative_method_input[1].setText(mfcc_params.get("derivative_method", ""))
            self.mfcc_width_input[1].setText(str(mfcc_params.get("sg_width", "")))
            self.mfcc_acc_order_input[1].setText(str(mfcc_params.get("fin_diff_acc_order", "")))
            self.mfcc_poly_order_input[1].setText(str(mfcc_params.get("sg_poly_order", "")))

        # Setting the Amplitude parameters
        amp_params = params.get("amplitude", {})
        if amp_params.get("enabled"):
            self.amp_enable_checkbox.setChecked(True)
            self.amp_method_input[1].setText(amp_params.get("method", ""))
            self.amp_winlen_input[1].setText(str(amp_params.get("winLen", "")))
            self.amp_hoplen_input[1].setText(str(amp_params.get("hopLen", "")))
            self.amp_center_input[1].setText("True" if amp_params.get("center") else "False")
            self.amp_outfilter_input[1].setText(amp_params.get("outFilter", "None"))
            self.amp_outfilt_type_input[1].setText(amp_params.get("outFiltType", ""))
            self.amp_outfilt_cutoff_input[1].setText(" ".join(map(str, amp_params.get("outFiltCutOff", []))))
            self.amp_outfilt_len_input[1].setText(str(amp_params.get("outFiltLen", "")))
            self.amp_outfilt_polyord_input[1].setText(str(amp_params.get("outFiltPolyOrd", "")))
            self.amp_name_input[1].setText(amp_params.get("name", ""))
            self.amp_panel_choice.setCurrentIndex(amp_params.get("panel", 0))
            self.amp_traj_radio.setChecked(amp_params.get("derivation_type") == 0)
            self.amp_vel_radio.setChecked(amp_params.get("derivation_type") == 1)
            self.amp_acc_radio.setChecked(amp_params.get("derivation_type") == 2)
            self.amp_derivative_method_input[1].setText(amp_params.get("derivative_method", ""))
            self.amp_width_input[1].setText(str(amp_params.get("sg_width", "")))
            self.amp_acc_order_input[1].setText(str(amp_params.get("fin_diff_acc_order", "")))
            self.amp_poly_order_input[1].setText(str(amp_params.get("sg_poly_order", "")))

        # Setting the Formant1 parameters
        formant1_params = params.get("formant1", {})
        if formant1_params.get("enabled"):
            self.formant1_enable_checkbox.setChecked(True)
            self.formant1_energy_threshold_input[1].setText(str(formant1_params.get("energy_threshold", "")))
            self.formant1_tstep_input[1].setText(str(formant1_params.get("time_step", "")))
            self.formant1_max_num_formants_input[1].setText(str(formant1_params.get("max_num_formants", "")))
            self.formant1_max_formant_input[1].setText(str(formant1_params.get("max_formant", "")))
            self.formant1_winlen_input[1].setText(str(formant1_params.get("window_length", "")))
            self.formant1_pre_emphasis_input[1].setText(str(formant1_params.get("pre_emphasis_from", "")))
            self.formant1_name_input[1].setText(formant1_params.get("name", ""))
            self.formant1_panel_choice.setCurrentIndex(formant1_params.get("panel", 0))
            self.formant1_traj_radio.setChecked(formant1_params.get("derivation_type") == 0)
            self.formant1_vel_radio.setChecked(formant1_params.get("derivation_type") == 1)
            self.formant1_acc_radio.setChecked(formant1_params.get("derivation_type") == 2)
            self.formant1_derivative_method_input[1].setText(formant1_params.get("derivative_method", ""))
            self.formant1_width_input[1].setText(str(formant1_params.get("sg_width", "")))
            self.formant1_acc_order_input[1].setText(str(formant1_params.get("fin_diff_acc_order", "")))
            self.formant1_poly_order_input[1].setText(str(formant1_params.get("sg_poly_order", "")))

        # Setting the Formant2 parameters
        formant2_params = params.get("formant2", {})
        if formant2_params.get("enabled"):
            self.formant2_enable_checkbox.setChecked(True)
            self.formant2_energy_threshold_input[1].setText(str(formant2_params.get("energy_threshold", "")))
            self.formant2_tstep_input[1].setText(str(formant2_params.get("time_step", "")))
            self.formant2_max_num_formants_input[1].setText(str(formant2_params.get("max_num_formants", "")))
            self.formant2_max_formant_input[1].setText(str(formant2_params.get("max_formant", "")))
            self.formant2_winlen_input[1].setText(str(formant2_params.get("window_length", "")))
            self.formant2_pre_emphasis_input[1].setText(str(formant2_params.get("pre_emphasis_from", "")))
            self.formant2_name_input[1].setText(formant2_params.get("name", ""))
            self.formant2_panel_choice.setCurrentIndex(formant2_params.get("panel", 0))
            self.formant2_traj_radio.setChecked(formant2_params.get("derivation_type") == 0)
            self.formant2_vel_radio.setChecked(formant2_params.get("derivation_type") == 1)
            self.formant2_acc_radio.setChecked(formant2_params.get("derivation_type") == 2)
            self.formant2_derivative_method_input[1].setText(formant2_params.get("derivative_method", ""))
            self.formant2_width_input[1].setText(str(formant2_params.get("sg_width", "")))
            self.formant2_acc_order_input[1].setText(str(formant2_params.get("fin_diff_acc_order", "")))
            self.formant2_poly_order_input[1].setText(str(formant2_params.get("sg_poly_order", "")))

        # Setting the Formant3 parameters
        formant3_params = params.get("formant3", {})
        if formant3_params.get("enabled"):
            self.formant3_enable_checkbox.setChecked(True)
            self.formant3_energy_threshold_input[1].setText(str(formant3_params.get("energy_threshold", "")))
            self.formant3_tstep_input[1].setText(str(formant3_params.get("time_step", "")))
            self.formant3_max_num_formants_input[1].setText(str(formant3_params.get("max_num_formants", "")))
            self.formant3_max_formant_input[1].setText(str(formant3_params.get("max_formant", "")))
            self.formant3_winlen_input[1].setText(str(formant3_params.get("window_length", "")))
            self.formant3_pre_emphasis_input[1].setText(str(formant3_params.get("pre_emphasis_from", "")))
            self.formant3_name_input[1].setText(formant3_params.get("name", ""))
            self.formant3_panel_choice.setCurrentIndex(formant3_params.get("panel", 0))
            self.formant3_traj_radio.setChecked(formant3_params.get("derivation_type") == 0)
            self.formant3_vel_radio.setChecked(formant3_params.get("derivation_type") == 1)
            self.formant3_acc_radio.setChecked(formant3_params.get("derivation_type") == 2)
            self.formant3_derivative_method_input[1].setText(formant3_params.get("derivative_method", ""))
            self.formant3_width_input[1].setText(str(formant3_params.get("sg_width", "")))
            self.formant3_acc_order_input[1].setText(str(formant3_params.get("fin_diff_acc_order", "")))
            self.formant3_poly_order_input[1].setText(str(formant3_params.get("sg_poly_order", "")))

        # Setting the F0 parameters
        f0_params = params.get("f0", {})
        if f0_params.get("enabled"):
            self.f0_enable_checkbox.setChecked(True)
            self.f0_method_input[1].setText(f0_params.get("method", ""))
            self.f0_hop_size_input[1].setText(str(f0_params.get("hopSize", "")))
            self.f0_min_pitch_input[1].setText(str(f0_params.get("minPitch", "")))
            self.f0_max_pitch_input[1].setText(str(f0_params.get("maxPitch", "")))
            self.f0_interp_unvoiced_input[1].setText(f0_params.get("interpUnvoiced", ""))
            self.f0_out_filter_input[1].setText(f0_params.get("outFilter", "None"))
            self.f0_out_filt_type_input[1].setText(f0_params.get("outFiltType", ""))
            self.f0_out_filt_cutoff_input[1].setText(" ".join(map(str, f0_params.get("outFiltCutOff", []))))
            self.f0_out_filt_len_input[1].setText(str(f0_params.get("outFiltLen", "")))
            self.f0_out_filt_polyord_input[1].setText(str(f0_params.get("outFiltPolyOrd", "")))
            self.f0_name_input[1].setText(f0_params.get("name", ""))
            self.f0_panel_choice.setCurrentIndex(f0_params.get("panel", 0))
            self.f0_traj_radio.setChecked(f0_params.get("derivation_type") == 0)
            self.f0_vel_radio.setChecked(f0_params.get("derivation_type") == 1)
            self.f0_acc_radio.setChecked(f0_params.get("derivation_type") == 2)
            self.f0_derivative_method_input[1].setText(f0_params.get("derivative_method", ""))
            self.f0_width_input[1].setText(str(f0_params.get("sg_width", "")))
            self.f0_acc_order_input[1].setText(str(f0_params.get("fin_diff_acc_order", "")))
            self.f0_poly_order_input[1].setText(str(f0_params.get("sg_poly_order", "")))
                    # Setting the EMA parameters
        # Setting the EMA parameters
        ema_params = params.get("ema", {})
        if ema_params:
            self.ema_target_sample_rate_input[1].setText(str(ema_params.get("target_sample_rate", "")))
            self.ema_derivative_method_input[1].setText(ema_params.get("derivative_method", ""))
            self.ema_width_input[1].setText(str(ema_params.get("sg_width", "")))
            self.ema_acc_order_input[1].setText(str(ema_params.get("fin_diff_acc_order", "")))
            self.ema_poly_order_input[1].setText(str(ema_params.get("sg_poly_order", "")))


    def toggle_mfcc_fields(self, state):
        enabled = state == QtCore.Qt.Checked
        for widget in [
            self.mfcc_sample_rate_input[1],
            self.mfcc_tstep_input[1],
            self.mfcc_winlen_input[1],
            self.mfcc_nmfcc_input[1],
            self.mfcc_nfft_input[1],
            self.mfcc_remove_first_input[1],
            self.mfcc_filt_cutoff_input[1],
            self.mfcc_filt_ord_input[1],
            self.mfcc_diff_method_input[1],
            self.mfcc_out_filter_input[1],
            self.mfcc_out_filt_type_input[1],
            self.mfcc_out_filt_cutoff_input[1],
            self.mfcc_out_filt_len_input[1],
            self.mfcc_out_filt_polyord_input[1],
            self.mfcc_name_input[1],
            self.mfcc_panel_choice
        ]:
            widget.setEnabled(enabled)
        self.mfcc_derivative_widget.setEnabled(enabled)
        self.mfcc_derivative_method_input[1].setEnabled(enabled)
        self.mfcc_width_input[1].setEnabled(enabled and (self.mfcc_vel_radio.isChecked() or self.mfcc_acc_radio.isChecked()))
        self.mfcc_acc_order_input[1].setEnabled(enabled and (self.mfcc_vel_radio.isChecked() or self.mfcc_acc_radio.isChecked()))
        self.mfcc_poly_order_input[1].setEnabled(enabled and (self.mfcc_vel_radio.isChecked() or self.mfcc_acc_radio.isChecked()))

    def toggle_amp_fields(self, state):
        enabled = state == QtCore.Qt.Checked
        for widget in [
            self.amp_method_input[1],
            self.amp_winlen_input[1],
            self.amp_hoplen_input[1],
            self.amp_center_input[1],
            self.amp_outfilter_input[1],
            self.amp_outfilt_type_input[1],
            self.amp_outfilt_cutoff_input[1],
            self.amp_outfilt_len_input[1],
            self.amp_outfilt_polyord_input[1],
            self.amp_name_input[1],
            self.amp_panel_choice
        ]:
            widget.setEnabled(enabled)
        self.amp_derivative_widget.setEnabled(enabled)
        self.amp_derivative_method_input[1].setEnabled(enabled)
        self.amp_width_input[1].setEnabled(enabled and (self.amp_vel_radio.isChecked() or self.amp_acc_radio.isChecked()))
        self.amp_acc_order_input[1].setEnabled(enabled and (self.amp_vel_radio.isChecked() or self.amp_acc_radio.isChecked()))
        self.amp_poly_order_input[1].setEnabled(enabled and (self.amp_vel_radio.isChecked() or self.amp_acc_radio.isChecked()))

    def toggle_formant1_fields(self, state):
        enabled = state == QtCore.Qt.Checked
        for widget in [
            self.formant1_energy_threshold_input[1],
            self.formant1_tstep_input[1],
            self.formant1_max_num_formants_input[1],
            self.formant1_max_formant_input[1],
            self.formant1_winlen_input[1],
            self.formant1_pre_emphasis_input[1],
            self.formant1_name_input[1],
            self.formant1_panel_choice
        ]:
            widget.setEnabled(enabled)
        self.formant1_derivative_widget.setEnabled(enabled)
        self.formant1_derivative_method_input[1].setEnabled(enabled)
        self.formant1_width_input[1].setEnabled(enabled and (self.formant1_vel_radio.isChecked() or self.formant1_acc_radio.isChecked()))
        self.formant1_acc_order_input[1].setEnabled(enabled and (self.formant1_vel_radio.isChecked() or self.formant1_acc_radio.isChecked()))
        self.formant1_poly_order_input[1].setEnabled(enabled and (self.formant1_vel_radio.isChecked() or self.formant1_acc_radio.isChecked()))

    def toggle_formant2_fields(self, state):
        enabled = state == QtCore.Qt.Checked
        for widget in [
            self.formant2_energy_threshold_input[1],
            self.formant2_tstep_input[1],
            self.formant2_max_num_formants_input[1],
            self.formant2_max_formant_input[1],
            self.formant2_winlen_input[1],
            self.formant2_pre_emphasis_input[1],
            self.formant2_name_input[1],
            self.formant2_panel_choice
        ]:
            widget.setEnabled(enabled)
        self.formant2_derivative_widget.setEnabled(enabled)
        self.formant2_derivative_method_input[1].setEnabled(enabled)
        self.formant2_width_input[1].setEnabled(enabled and (self.formant2_vel_radio.isChecked() or self.formant2_acc_radio.isChecked()))
        self.formant2_acc_order_input[1].setEnabled(enabled and (self.formant2_vel_radio.isChecked() or self.formant2_acc_radio.isChecked()))
        self.formant2_poly_order_input[1].setEnabled(enabled and (self.formant2_vel_radio.isChecked() or self.formant2_acc_radio.isChecked()))

    def toggle_formant3_fields(self, state):
        enabled = state == QtCore.Qt.Checked
        for widget in [
            self.formant3_energy_threshold_input[1],
            self.formant3_tstep_input[1],
            self.formant3_max_num_formants_input[1],
            self.formant3_max_formant_input[1],
            self.formant3_winlen_input[1],
            self.formant3_pre_emphasis_input[1],
            self.formant3_name_input[1],
            self.formant3_panel_choice
        ]:
            widget.setEnabled(enabled)
        self.formant3_derivative_widget.setEnabled(enabled)
        self.formant3_derivative_method_input[1].setEnabled(enabled)
        self.formant3_width_input[1].setEnabled(enabled and (self.formant3_vel_radio.isChecked() or self.formant3_acc_radio.isChecked()))
        self.formant3_acc_order_input[1].setEnabled(enabled and (self.formant3_vel_radio.isChecked() or self.formant3_acc_radio.isChecked()))
        self.formant3_poly_order_input[1].setEnabled(enabled and (self.formant3_vel_radio.isChecked() or self.formant3_acc_radio.isChecked()))

    def toggle_f0_fields(self, state):
        enabled = state == QtCore.Qt.Checked
        for widget in [
            self.f0_method_input[1],
            self.f0_hop_size_input[1],
            self.f0_min_pitch_input[1],
            self.f0_max_pitch_input[1],
            self.f0_interp_unvoiced_input[1],
            self.f0_out_filter_input[1],
            self.f0_out_filt_type_input[1],
            self.f0_out_filt_cutoff_input[1],
            self.f0_out_filt_len_input[1],
            self.f0_out_filt_polyord_input[1],
            self.f0_name_input[1],
            self.f0_panel_choice
        ]:
            widget.setEnabled(enabled)
        self.f0_derivative_widget.setEnabled(enabled)
        self.f0_derivative_method_input[1].setEnabled(enabled)
        self.f0_width_input[1].setEnabled(enabled and (self.f0_vel_radio.isChecked() or self.f0_acc_radio.isChecked()))
        self.f0_acc_order_input[1].setEnabled(enabled and (self.f0_vel_radio.isChecked() or self.f0_acc_radio.isChecked()))
        self.f0_poly_order_input[1].setEnabled(enabled and (self.f0_vel_radio.isChecked() or self.f0_acc_radio.isChecked()))

    def toggle_mfcc_derivative_fields(self):
        enabled = self.mfcc_vel_radio.isChecked() or self.mfcc_acc_radio.isChecked()
        self.mfcc_derivative_method_input[1].setEnabled(enabled)
        self.mfcc_width_input[1].setEnabled(enabled)
        self.mfcc_acc_order_input[1].setEnabled(enabled)
        self.mfcc_poly_order_input[1].setEnabled(enabled)

    def toggle_amp_derivative_fields(self):
        enabled = self.amp_vel_radio.isChecked() or self.amp_acc_radio.isChecked()
        self.amp_derivative_method_input[1].setEnabled(enabled)
        self.amp_width_input[1].setEnabled(enabled)
        self.amp_acc_order_input[1].setEnabled(enabled)
        self.amp_poly_order_input[1].setEnabled(enabled)

    def toggle_formant1_derivative_fields(self):
        enabled = self.formant1_vel_radio.isChecked() or self.formant1_acc_radio.isChecked()
        self.formant1_derivative_method_input[1].setEnabled(enabled)
        self.formant1_width_input[1].setEnabled(enabled)
        self.formant1_acc_order_input[1].setEnabled(enabled)
        self.formant1_poly_order_input[1].setEnabled(enabled)

    def toggle_formant2_derivative_fields(self):
        enabled = self.formant2_vel_radio.isChecked() or self.formant2_acc_radio.isChecked()
        self.formant2_derivative_method_input[1].setEnabled(enabled)
        self.formant2_width_input[1].setEnabled(enabled)
        self.formant2_acc_order_input[1].setEnabled(enabled)
        self.formant2_poly_order_input[1].setEnabled(enabled)

    def toggle_formant3_derivative_fields(self):
        enabled = self.formant3_vel_radio.isChecked() or self.formant3_acc_radio.isChecked()
        self.formant3_derivative_method_input[1].setEnabled(enabled)
        self.formant3_width_input[1].setEnabled(enabled)
        self.formant3_acc_order_input[1].setEnabled(enabled)
        self.formant3_poly_order_input[1].setEnabled(enabled)

    def toggle_f0_derivative_fields(self):
        enabled = self.f0_vel_radio.isChecked() or self.f0_acc_radio.isChecked()
        self.f0_derivative_method_input[1].setEnabled(enabled)
        self.f0_width_input[1].setEnabled(enabled)
        self.f0_acc_order_input[1].setEnabled(enabled)
        self.f0_poly_order_input[1].setEnabled(enabled)
