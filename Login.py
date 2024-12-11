import sys
import pyodbc
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QCheckBox, QMessageBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

class LoginApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Login/Register")

        screen = QApplication.primaryScreen().geometry()
        self.setFixedSize(screen.width() // 2, 700)
        self.setStyleSheet("background-color: white;")

        title_font = QFont("Arial", 20, QFont.Bold)
        label_font = QFont("Arial", 10)
        button_font = QFont("Arial", 12)

        self.left_frame = QFrame()
        self.left_frame.setStyleSheet("background-color: white; padding: 20px;")

        login_title = QLabel("Login")
        login_title.setFont(title_font)
        login_title.setAlignment(Qt.AlignCenter)
        login_title.setStyleSheet("color: #0078D7; padding: 5px;")

        login_description = QLabel("Accessing this course requires a login, please enter your credentials below!")
        login_description.setFont(label_font)
        login_description.setWordWrap(True)
        login_description.setAlignment(Qt.AlignCenter)

        self.username_label = QLabel("Username or Email Address")
        self.username_label.setFont(label_font)
        self.username_input = QLineEdit()
        self.username_input.setFont(label_font)
        self.username_input.setPlaceholderText("Enter your username or email")

        self.password_label = QLabel("Password")
        self.password_label.setFont(label_font)
        self.password_input = QLineEdit()
        self.password_input.setFont(label_font)
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setEchoMode(QLineEdit.Password)

        remember_me_checkbox = QCheckBox("Remember Me")
        remember_me_checkbox.setFont(label_font)

        self.login_button = QPushButton("Log In")
        self.login_button.setFont(button_font)
        self.login_button.setStyleSheet(
            "background-color: #0078D7; color: white; padding: 5px; border-radius: 5px;"
        )
        self.login_button.clicked.connect(self.login)

        left_layout = QVBoxLayout()
        left_layout.addWidget(login_title)
        left_layout.addWidget(login_description)
        left_layout.addWidget(self.username_label)
        left_layout.addWidget(self.username_input)
        left_layout.addWidget(self.password_label)
        left_layout.addWidget(self.password_input)
        left_layout.addWidget(remember_me_checkbox)
        left_layout.addWidget(self.login_button)
        left_layout.addStretch()
        self.left_frame.setLayout(left_layout)

        self.right_frame = QFrame()
        self.right_frame.setStyleSheet("background-color: #0078D7; padding: 20px;")

        register_title = QLabel("Register")
        register_title.setFont(title_font)
        register_title.setAlignment(Qt.AlignCenter)
        register_title.setStyleSheet("color: white;")

        register_description = QLabel("Don't have an account? Register one!")
        register_description.setFont(label_font)
        register_description.setAlignment(Qt.AlignCenter)
        register_description.setStyleSheet("color: white;")

        self.register_button = QPushButton("Register an Account")
        self.register_button.setFont(button_font)
        self.register_button.setStyleSheet(
            "background-color: white; color: #0078D7; padding: 5px; border-radius: 5px;"
        )
        self.register_button.clicked.connect(self.show_reg)

        right_layout = QVBoxLayout()
        right_layout.addWidget(register_title)
        right_layout.addWidget(register_description)
        right_layout.addWidget(self.register_button)
        right_layout.addStretch()
        self.right_frame.setLayout(right_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.left_frame)
        main_layout.addWidget(self.right_frame)

        self.setLayout(main_layout)

        self.conn = None
        self.cursor = None
        self.setup_db_connection()

    def setup_db_connection(self):
        try:
            self.conn = pyodbc.connect(
                r"DRIVER={ODBC Driver 17 for SQL Server};"
                r"SERVER=localhost\MSSQLSERVER01;"
                r"DATABASE=pythonlogin;"
                r"Trusted_Connection=yes;"
            )
            self.cursor = self.conn.cursor()
            print("Successful connection!")
        except Exception as e:
            print(f"Connection error: {e}")

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if not username or not password:
            QMessageBox.warning(self, "Error", "Please enter both username and password.")
            return

        query = "SELECT * FROM accounts WHERE username = ? AND password = ?"
        self.cursor.execute(query, (username, password))

        if self.cursor.fetchone():
            QMessageBox.information(self, "Success", "Login successful!")
            self.close()  # Close LoginApp
            self.open_welcome_page()  # Open WelcomePage
        else:
            QMessageBox.warning(self, "Error", "Invalid username or password.")

    def open_welcome_page(self):
        from WelcomePage import Welcome
        self.welcome_window = Welcome()
        self.welcome_window.show()

    def show_reg(self):
        from Register import RegistrationApp
        self.close()
        self.registration_window = RegistrationApp()
        self.registration_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoginApp()
    window.show()
    sys.exit(app.exec_())

