import sys
import pyodbc
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QMessageBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt


class RegistrationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("User Registration")

        screen = QApplication.primaryScreen().geometry()
        self.setFixedSize(screen.width() // 2, 700)
        self.setStyleSheet("background-color: white;")

        title_font = QFont("Arial", 20, QFont.Bold)
        label_font = QFont("Arial", 10)
        button_font = QFont("Arial", 12)

        self.left_frame = QFrame()
        self.left_frame.setStyleSheet("background-color: white; padding: 20px;")

        register_title = QLabel("Register")
        register_title.setFont(title_font)
        register_title.setAlignment(Qt.AlignCenter)
        register_title.setStyleSheet("color: #0078D7; padding: 5px;")

        register_description = QLabel("Create a new account by filling in the fields below!")
        register_description.setFont(label_font)
        register_description.setWordWrap(True)
        register_description.setAlignment(Qt.AlignCenter)

        self.username_label = QLabel("Username")
        self.username_label.setFont(label_font)
        self.username_input = QLineEdit()
        self.username_input.setFont(label_font)
        self.username_input.setPlaceholderText("Enter your username")

        self.email_label = QLabel("Email Address")
        self.email_label.setFont(label_font)
        self.email_input = QLineEdit()
        self.email_input.setFont(label_font)
        self.email_input.setPlaceholderText("Enter your email")

        self.password_label = QLabel("Password")
        self.password_label.setFont(label_font)
        self.password_input = QLineEdit()
        self.password_input.setFont(label_font)
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setEchoMode(QLineEdit.Password)

        self.register_button = QPushButton("Register")
        self.register_button.setFont(button_font)
        self.register_button.setStyleSheet(
            "background-color: #0078D7; color: white; padding: 5px; border-radius: 5px;"
        )
        self.register_button.clicked.connect(self.register)

        left_layout = QVBoxLayout()
        left_layout.addWidget(register_title)
        left_layout.addWidget(register_description)
        left_layout.addWidget(self.username_label)
        left_layout.addWidget(self.username_input)
        left_layout.addWidget(self.email_label)
        left_layout.addWidget(self.email_input)
        left_layout.addWidget(self.password_label)
        left_layout.addWidget(self.password_input)
        left_layout.addWidget(self.register_button)
        left_layout.addStretch()
        self.left_frame.setLayout(left_layout)

        self.right_frame = QFrame()
        self.right_frame.setStyleSheet("background-color: #0078D7; padding: 20px;")

        info_title = QLabel("Welcome")
        info_title.setFont(title_font)
        info_title.setAlignment(Qt.AlignCenter)
        info_title.setStyleSheet("color: white;")

        info_description = QLabel("Already have an account? Log in to access your dashboard.")
        info_description.setFont(label_font)
        info_description.setAlignment(Qt.AlignCenter)
        info_description.setStyleSheet("color: white;")

        self.login_button = QPushButton("Log In")
        self.login_button.setFont(button_font)
        self.login_button.setStyleSheet(
            "background-color: white; color: #0078D7; padding: 5px; border-radius: 5px;"
        )
        self.login_button.clicked.connect(self.show_login)

        right_layout = QVBoxLayout()
        right_layout.addWidget(info_title)
        right_layout.addWidget(info_description)
        right_layout.addWidget(self.login_button)
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
            print("Database connection successful!")
        except Exception as e:
            print(f"Database connection error: {e}")

    def register(self):
        username = self.username_input.text()
        email = self.email_input.text()
        password = self.password_input.text()

        if not username or not email or not password:
            self.show_message("Error", "Please fill in all fields.")
            return

        query_check = "SELECT * FROM accounts WHERE username = ? OR email = ?"
        self.cursor.execute(query_check, (username, email))
        if self.cursor.fetchone():
            self.show_message("Error", "Username or email already exists.")
            self.username_input.clear()
            self.email_input.clear()
            self.password_input.clear()
            return

        query_insert = "INSERT INTO accounts (username, email, password) VALUES (?, ?, ?)"
        self.cursor.execute(query_insert, (username, email, password))
        self.conn.commit()

        self.show_success_message("Success", "Registration successful!")
        self.show_login()

    def show_success_message(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def show_message(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def show_login(self):
        from Login import LoginApp
        self.close()
        self.login_window = LoginApp()
        self.login_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RegistrationApp()
    window.show()
    sys.exit(app.exec_())