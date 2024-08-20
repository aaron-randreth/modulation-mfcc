
from PyQt5 import QtWidgets, QtCore
class UnifiedConfigDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Parameters")
        self.layout = QtWidgets.QGridLayout()

        # Main layout with a scroll area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QGridLayout(scroll_content)
        scroll_area.setWidget(scroll_content)

        # Checkbox for enabling/disabling MFCC customization
        self.mfcc_enable_checkbox = QtWidgets.QCheckBox("Enable MFCC Customization")
        self.mfcc_enable_checkbox.setChecked(False)
        self.mfcc_enable_checkbox.stateChanged.connect(self.toggle_mfcc_fields)

        # MFCC Configuration
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

        self.mfcc_derivative_group.setExclusive(True)

        # Add derivatives section
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
        
        self.mfcc_derivative_group.buttonClicked.connect(self.toggle_mfcc_derivative_fields)

        self.toggle_mfcc_fields(self.mfcc_enable_checkbox.checkState())

        # Checkbox for enabling/disabling Amplitude customization
        self.amp_enable_checkbox = QtWidgets.QCheckBox("Enable Amplitude Customization")
        self.amp_enable_checkbox.setChecked(False)
        self.amp_enable_checkbox.stateChanged.connect(self.toggle_amp_fields)

        # Amplitude Configuration
        self.amp_method_input = self.create_input_field("Method (RMS/Hilb/RMSpraat):", "RMS")
        self.amp_winlen_input = self.create_input_field("Window Length (s):", "0.1")
        self.amp_hoplen_input = self.create_input_field("Hop Length (s):", "0.01")
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
        self.amp_derivative_group.setExclusive(True)

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

        self.amp_derivative_group.buttonClicked.connect(self.toggle_amp_derivative_fields)
        self.toggle_amp_derivative_fields()

        # Checkbox for enabling/disabling Formant1 customization
        self.formant1_enable_checkbox = QtWidgets.QCheckBox("Enable Formant1 Customization")
        self.formant1_enable_checkbox.setChecked(False)
        self.formant1_enable_checkbox.stateChanged.connect(self.toggle_formant1_fields)

        # Formant1 Configuration
        self.formant1_energy_threshold_input = self.create_input_field("Energy Threshold:", "20.0")
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
        self.formant1_derivative_group.setExclusive(True)

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

        self.formant1_derivative_group.buttonClicked.connect(self.toggle_formant1_derivative_fields)
        self.toggle_formant1_derivative_fields()

        # Checkbox for enabling/disabling Formant2 customization
        self.formant2_enable_checkbox = QtWidgets.QCheckBox("Enable Formant2 Customization")
        self.formant2_enable_checkbox.setChecked(False)
        self.formant2_enable_checkbox.stateChanged.connect(self.toggle_formant2_fields)

        # Formant2 Configuration
        self.formant2_energy_threshold_input = self.create_input_field("Energy Threshold:", "20.0")
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
        self.formant2_derivative_group.setExclusive(True)

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

        self.formant2_derivative_group.buttonClicked.connect(self.toggle_formant2_derivative_fields)
        self.toggle_formant2_derivative_fields()

        # Checkbox for enabling/disabling Formant3 customization
        self.formant3_enable_checkbox = QtWidgets.QCheckBox("Enable Formant3 Customization")
        self.formant3_enable_checkbox.setChecked(False)
        self.formant3_enable_checkbox.stateChanged.connect(self.toggle_formant3_fields)

        # Formant3 Configuration
        self.formant3_energy_threshold_input = self.create_input_field("Energy Threshold:", "20.0")
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
        self.formant3_derivative_group.setExclusive(True)

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

        self.formant3_derivative_group.buttonClicked.connect(self.toggle_formant3_derivative_fields)
        self.toggle_formant3_derivative_fields()

        # F0 Configuration
        self.f0_enable_checkbox = QtWidgets.QCheckBox("Enable F0 Customization")

        self.f0_enable_checkbox.setChecked(False)
        self.f0_enable_checkbox.stateChanged.connect(self.toggle_f0_fields)
        self.f0_method_input = self.create_input_field("Method (praatac/pyin):", "praatac")
        self.f0_hop_size_input = self.create_input_field("Hop Size (s):", "0.01")
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
        self.f0_derivative_group.setExclusive(True)

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

        self.f0_derivative_group.buttonClicked.connect(self.toggle_f0_derivative_fields)
        self.toggle_f0_derivative_fields()


        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.clicked.connect(self.accept)
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
        ], scroll_layout, 0, 0)

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
        ], scroll_layout, 0, 1)

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
        ], scroll_layout, 0, 2)

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
        ], scroll_layout, 1, 0)

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
        ], scroll_layout, 1, 1)

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
        ], scroll_layout, 1, 2)

        scroll_layout.addWidget(self.apply_button, 2, 0, 1, 3)

        self.layout.addWidget(scroll_area)
        self.setLayout(self.layout)
        self.toggle_mfcc_derivative_fields()

        self.toggle_amp_fields(self.amp_enable_checkbox.checkState())
        self.toggle_formant1_fields(self.formant1_enable_checkbox.checkState())
        self.toggle_formant2_fields(self.formant2_enable_checkbox.checkState())
        self.toggle_formant3_fields(self.formant3_enable_checkbox.checkState())
        self.toggle_f0_fields(self.f0_enable_checkbox.checkState())
    def create_input_field(self, label_text, default_value):
        label = QtWidgets.QLabel(label_text)
        input_field = QtWidgets.QLineEdit(default_value)
        container = QtWidgets.QVBoxLayout()
        container.addWidget(label)
        container.addWidget(input_field)
        container.setSpacing(1)  # Reduce spacing between label and input field
        container.setContentsMargins(1, 1, 1, 1)  # Reduce margins inside the container
        widget = QtWidgets.QWidget()
        widget.setLayout(container)
        return widget, input_field

    def add_groupbox_to_layout(self, title, widgets, layout, row, col):
        group_box = QtWidgets.QGroupBox(title)
        group_box_layout = QtWidgets.QVBoxLayout()
        group_box_layout.setSpacing(1)  # Reduce spacing between widgets in the group box
        group_box_layout.setContentsMargins(1, 1, 1, 1)  # Reduce margins inside the group box
        group_box.setLayout(group_box_layout)

        for widget in widgets:
            if isinstance(widget, tuple):
                h_layout = QtWidgets.QHBoxLayout()
                h_layout.addWidget(widget[0])
                h_layout.addWidget(widget[1])
                h_layout.setSpacing(1)  # Reduce spacing between label and combo box
                container = QtWidgets.QWidget()
                container.setLayout(h_layout)
                group_box_layout.addWidget(container)
            else:
                group_box_layout.addWidget(widget)

        layout.addWidget(group_box, row, col)

    def get_parameters(self):
        mfcc_enabled = self.mfcc_enable_checkbox.isChecked()
        amp_enabled = self.amp_enable_checkbox.isChecked()
        formant1_enabled = self.formant1_enable_checkbox.isChecked()
        formant2_enabled = self.formant2_enable_checkbox.isChecked()
        formant3_enabled = self.formant3_enable_checkbox.isChecked()
        f0_enabled = self.f0_enable_checkbox.isChecked()

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
            }
        }
        return params

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