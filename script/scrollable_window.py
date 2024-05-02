import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QFrame,
    QVBoxLayout,
    QScrollArea,
    QLabel,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt


class Output(QLabel):
    label: str
    unit: str

    def __init__(self, label: str, unit: str) -> None:
        super().__init__()

        self.label = label
        self.unit = unit

    def update(self, value: float | int) -> None:
        self.setText(f"<b><u>{self.label}</u></b>: {value} {self.unit}")


class InfoBox(QFrame):
    title: str
    paragraph: str | None
    dynamic_content: list[Output] | None

    def __init__(
        self,
        title: str,
        paragraph: str | None = None,
        dynamic_content: list[Output] | None = None,
    ):
        super().__init__()

        self.title = title
        self.paragraph = paragraph

        if dynamic_content is None:
            dynamic_content = []

        self.dynamic_content = dynamic_content

        self.init_ui()

    def create_title(self) -> QHBoxLayout:
        title_label = QLabel(self.title)
        title_label.setStyleSheet("font-weight: bold;")

        title_layout = QHBoxLayout()
        title_layout.addWidget(title_label, alignment=Qt.AlignCenter)

        return title_layout

    def set_style(self) -> None:
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setLineWidth(1)
        self.setStyleSheet("background-color: white")
        self.setMinimumHeight(100)

    def init_ui(self) -> None:
        self.set_style()
        layout = QVBoxLayout()

        title_layout = self.create_title()
        layout.addLayout(title_layout)

        if self.paragraph is not None:
            paragraph_label = QLabel(self.paragraph)
            layout.addWidget(paragraph_label)

        self.handle_dynamic_content(layout)

        self.setLayout(layout)

    def handle_dynamic_content(self, layout: QVBoxLayout) -> None:
        if self.dynamic_content is None:
            return

        for content in self.dynamic_content:
            layout.addWidget(content)


class Info(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.setMaximumWidth(400)

        layout = QVBoxLayout()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)

        self.scroll_area.setWidget(scroll_content)
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)

    def add_infobox(self, box: InfoBox) -> None:
        self.scroll_layout.addWidget(box)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    info_widget = Info()

    output = Output("Mesures", "")
    box = InfoBox("Calcul MFCC", dynamic_content=output)
    info_widget.add_infobox(box)
    output.update(12.22)

    info_widget.show()
    sys.exit(app.exec_())
