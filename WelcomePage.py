from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget

from Dashboard import Dashboard


class Welcome(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        screen = QApplication.primaryScreen().geometry()
        self.setFixedSize(screen.width() // 2, 700)

    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Hurricane Dashboard")
        title.setStyleSheet(
            """
            font-size: 32px;
            font-weight: bold;
            color: #005BBB;
            text-align: center;
            """
        )
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        description = QLabel("Welcome! This is an application for analyzing and visualizing hurricane data.")
        description.setStyleSheet(
            """
            font-size: 16px;
            color: #004080;
            text-align: center;
            """
        )
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignCenter)
        layout.addWidget(description)

        img = QLabel()
        pixmap = QPixmap("Visual data-bro.png")

        scaled_pixmap = pixmap.scaled(
            self.width() - 40,
            self.height() // 2,
            Qt.KeepAspectRatio
        )
        img.setPixmap(scaled_pixmap)
        img.setAlignment(Qt.AlignCenter)
        img.setStyleSheet("margin-top: 20px; margin-bottom: 20px;")
        layout.addWidget(img)

        btn_start = QPushButton("Start the analysis")
        btn_start.setStyleSheet(
            """
            QPushButton {
                background-color: #0078D7;
                color: white;
                padding: 12px 24px;
                font-size: 18px;
                font-weight: bold;
                border-radius: 10px;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: #005BBB;
            }
            QPushButton:pressed {
                background-color: #003F7F;
            }
            """
        )
        btn_start.clicked.connect(self.goto_dashboard)
        btn_start.setFixedWidth(250)
        layout.addWidget(btn_start, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def goto_dashboard(self):
        self.close()
        self.dashboard_window = Dashboard()
        self.dashboard_window.show()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Hurricane Dashboard")
        screen = QApplication.primaryScreen().geometry()
        self.setFixedSize(screen.width() // 2, 700)

        self.setStyleSheet("background-color: #EAF3FF;")
        self.setCentralWidget(Welcome())


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
