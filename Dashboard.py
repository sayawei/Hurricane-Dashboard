import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QLabel, QMenuBar, QAction, QWidget, QScrollArea,
    QTableWidget, QTableWidgetItem, QPushButton, QStackedWidget, QLineEdit
)
import folium
from PyQt5.QtGui import QPixmap
from minisom import MiniSom
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from PyQt5.QtWebEngineWidgets import QWebEngineView
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class HurricaneMap(QWidget):
    def __init__(self, hurricane_data, dashboard):
        super().__init__()
        self.dashboard = dashboard
        self.layout = QVBoxLayout(self)

        self.map = folium.Map(location=[20, -80], zoom_start=4)

        for index, row in hurricane_data.iterrows():
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=f"Cyclone Category: {row['CycloneCategory']}",
                icon=folium.Icon(color='red')
            ).add_to(self.map)

        self.map.save('hurricane_map.html')

        self.web_view = QWebEngineView()
        self.web_view.setUrl(QUrl.fromLocalFile(os.path.abspath('hurricane_map.html')))
        self.layout.addWidget(self.web_view)

        back_button = QPushButton("Back to Graphs", self)
        back_button.clicked.connect(self.dashboard.show_graph_page)
        self.layout.addWidget(back_button)

class DataInfo(QWidget):
    def __init__(self, data, dashboard):
        super().__init__()
        self.dashboard = dashboard
        self.layout = QVBoxLayout(self)

        self.title = QLabel("Data Information")
        self.title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title)

        info_label = QLabel("This dashboard provides insights into hurricane data, including visualizations and predictions.", self)
        info_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(info_label)

        self.search_box = QLineEdit(self)
        self.search_box.setPlaceholderText("Search...")
        self.search_box.textChanged.connect(self.filter_table)
        self.layout.addWidget(self.search_box)

        self.table_widget = QTableWidget()
        self.layout.addWidget(self.table_widget)

        self.populate_table(data)

        back_button = QPushButton("Back to Graphs", self)
        back_button.clicked.connect(self.dashboard.show_graph_page)
        self.layout.addWidget(back_button)
    def populate_table(self, data):
        self.table_widget.setRowCount(len(data))
        self.table_widget.setColumnCount(len(data.columns))
        self.table_widget.setHorizontalHeaderLabels(data.columns)

        for row in range(len(data)):
            for column in range(len(data.columns)):
                self.table_widget.setItem(row, column, QTableWidgetItem(str(data.iat[row, column])))

    def filter_table(self):
        search_text = self.search_box.text().lower()
        for row in range(self.table_widget.rowCount()):
            item = self.table_widget.item(row, 0)
            if item and search_text in item.text().lower():
                self.table_widget.showRow(row)
            else:
                self.table_widget.hideRow(row)
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hurricane Dashboard")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        file_menu = self.menu_bar.addMenu("Menu")
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        hurricane_map_action = QAction("Hurricane Map", self)
        hurricane_map_action.triggered.connect(self.show_hurricane_map_page)
        file_menu.addAction(hurricane_map_action)

        info_action = QAction("Data Info", self)
        info_action.triggered.connect(self.show_data_info_page)
        file_menu.addAction(info_action)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.stacked_widget = QStackedWidget(self)
        self.layout.addWidget(self.stacked_widget)

        self.data = pd.read_csv("hurricane/Hurricane_Data.csv")

        self.start_date = "2024-10-05"
        self.end_date = "2024-12-05"
        self.filtered_data = self.data[
            (self.data['Timestamp'] >= self.start_date) & (self.data['Timestamp'] <= self.end_date)
        ]

        self.create_graph_page()
        self.create_filtered_data_page()
        self.create_hurricane_map_page()
        self.create_data_info_page()

        self.generate_plots()

        self.predict_hurricane_intensity()

    def create_graph_page(self):
        self.graph_page = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_page)

        self.title = QLabel("Hurricane Data Visualization")
        self.title.setAlignment(Qt.AlignCenter)
        self.graph_layout.addWidget(self.title)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.graph_layout.addWidget(self.scroll_area)

        self.scroll_content = QWidget()
        self.scroll_area.setWidget(self.scroll_content)

        self.scroll_layout = QVBoxLayout(self.scroll_content)

        self.stacked_widget.addWidget(self.graph_page)

    def populate_table(self, data):
        self.table_widget.setRowCount(data.shape[0])
        self.table_widget.setColumnCount(data.shape[1])
        self.table_widget.setHorizontalHeaderLabels(data.columns)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                self.table_widget.setItem(i, j, QTableWidgetItem(str(data.iat[i, j])))


        self.table_widget.resizeColumnsToContents()
        self.table_widget.resizeRowsToContents()
    def create_filtered_data_page(self):
        self.filtered_data_page = QWidget()
        self.filtered_data_layout = QVBoxLayout(self.filtered_data_page)

        self.filtered_data_title = QLabel("Hurricane Data")
        self.filtered_data_title.setAlignment(Qt.AlignCenter)
        self.filtered_data_layout.addWidget(self.filtered_data_title)

        self.data_table = QTableWidget(self)
        self.filtered_data_layout.addWidget(self.data_table)

        self.back_button = QPushButton("Back to Graphs", self)
        self.back_button.clicked.connect(self.show_graph_page)
        self.filtered_data_layout.addWidget(self.back_button)

        self.stacked_widget.addWidget(self.filtered_data_page)

        self.update_filtered_data_table()

    def update_filtered_data_table(self):
        self.data_table.setRowCount(len(self.filtered_data))
        self.data_table.setColumnCount(len(self.filtered_data.columns))
        self.data_table.setHorizontalHeaderLabels(self.filtered_data.columns)

        for row in range(len(self.filtered_data)):
            for column in range(len(self.filtered_data.columns)):
                self.data_table.setItem(row, column, QTableWidgetItem(str(self.filtered_data.iat[row, column])))

    def show_graph_page(self):
        self.stacked_widget.setCurrentIndex(0)

    def create_hurricane_map_page(self):
        self.hurricane_map = HurricaneMap(self.filtered_data, self)
        self.stacked_widget.addWidget(self.hurricane_map)

    def create_data_info_page(self):
        self.data_info = DataInfo(self.data, self)
        self.stacked_widget.addWidget(self.data_info)
    def show_hurricane_map_page(self):
        self.stacked_widget.setCurrentIndex(2)

    def show_data_info_page(self):
        self.stacked_widget.setCurrentIndex(3)

    def generate_plots(self):
        self.create_line_plot('Wind Speed Over Time', 'WindSpeed_kmh')
        self.create_bar_plot('Average Wind Speed by Month', 'WindSpeed_kmh')
        self.create_scatter_plot('Wind Speed vs Atmospheric Pressure', 'WindSpeed_kmh', 'AtmosphericPressure_hPa')

    def create_line_plot(self, title, column):
        plot_filename = f'images/{title.replace(" ", "_").lower()}.png'
        if not os.path.exists(plot_filename):
            plt.figure(figsize=(10, 5))
            plt.plot(self.filtered_data['Timestamp'], self.filtered_data[column], label=title, color='blue')
            plt.title(title)
            plt.xlabel('Time')
            plt.ylabel(title)
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_filename)
            plt.close()

        self.display_plot(plot_filename, title)

    def create_bar_plot(self, title, column):
        plot_filename = f'images/{title.replace(" ", "_").lower()}.png'
        if not os.path.exists(plot_filename):
            plt.figure (figsize=(10, 5))
            self.filtered_data['Month'] = pd.to_datetime(self.filtered_data['Timestamp']).dt.month
            monthly_avg = self.filtered_data.groupby('Month')[column].mean()
            monthly_avg.plot(kind='bar', color='orange')
            plt.title(title)
            plt.xlabel('Month')
            plt.ylabel('Average ' + title)
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(plot_filename)
            plt.close()

        self.display_plot(plot_filename, title)

    def create_scatter_plot(self, title, x_column, y_column):
        plot_filename = f'images/{title.replace(" ", "_").lower()}.png'
        if not os.path.exists(plot_filename):
            plt.figure(figsize=(10, 5))
            plt.scatter(self.filtered_data[x_column], self.filtered_data[y_column], alpha=0.5)
            plt.title(title)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_filename)
            plt.close()

        self.display_plot(plot_filename, title)

    def classify_hurricanes(csv_file, new_example=None):
        data = pd.read_csv(csv_file)

        X = data[['WindSpeed_kmh', 'AtmosphericPressure_hPa', 'Humidity_%']]
        y = data['CycloneCategory']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        report = classification_report(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        print("Classification Report:")
        print(report)
        print("Accuracy:", accuracy)

        if new_example is not None:
            new_example_scaled = scaler.transform(new_example)
            predicted_category = model.predict(new_example_scaled)
            print("Predicted Cyclone Category:", predicted_category)

    def display_plot(self, plot_filename, title):
        plot_label = QLabel(self)
        pixmap = QPixmap(plot_filename)
        plot_label.setPixmap(pixmap)
        plot_label.setAlignment(Qt.AlignCenter)

        description_label = QLabel(f"<b>{title}</b>", self)
        description_label.setAlignment(Qt.AlignCenter)

        self.scroll_layout.addWidget(description_label)
        self.scroll_layout.addWidget(plot_label)

    def create_kohonen_map(self):
        plot_filename = 'images/kohonen_map.png'
        if not os.path.exists(plot_filename):
            data_for_som = self.filtered_data[['WindSpeed_kmh', 'AtmosphericPressure_hPa']].values

            som_size = 10
            som = MiniSom(som_size, som_size, 2, sigma=1.0, learning_rate=0.5)
            som.train(data_for_som, 1000)

            win_map = som.win_map(data_for_som)

            plt.figure(figsize=(8, 8))
            for x in range(som_size):
                for y in range(som_size):
                    if win_map[(x, y)]:
                        plt.scatter(x, y, marker='o', color='blue', alpha=0.5)

            plt.title('Kohonen Map of Hurricane Data')
            plt.xlabel('SOM X')
            plt.ylabel('SOM Y')
            plt.grid(True)

            plt.savefig(plot_filename)
            plt.close()

        self.display_plot(plot_filename, 'Kohonen Map')

    def predict_hurricane_intensity(self):
        plot_filename_rf = 'images/predicted_hurricane_intensity_rf.png'
        plot_filename_lr = 'images/predicted_hurricane_intensity_lr.png'
        plot_filename_logreg = 'images/predicted_hurricane_intensity_logreg.png'
        plot_filename_svm = 'images/predicted_hurricane_intensity_svm.png'

        if 'CycloneCategory' not in self.filtered_data.columns:
            print("The 'CycloneCategory' column is missing from the data.")
            return

        X = self.filtered_data[
            ['WindSpeed_kmh', 'AtmosphericPressure_hPa', 'AirTemperature_C']]
        y = self.filtered_data['CycloneCategory']

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)
        self.visualize_rf_predictions(X_test, y_test, rf_predictions, plot_filename_rf)

        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_predictions = lr_model.predict(X_test)
        lr_predictions = [round(pred) for pred in lr_predictions]
        self.visualize_lr_distribution(lr_predictions, plot_filename_lr)

        logreg_model = LogisticRegression(max_iter=1000)
        logreg_model.fit(X_train, y_train)
        logreg_predictions = logreg_model.predict(X_test)
        self.visualize_logreg_scatter(X_test, y_test, logreg_predictions, plot_filename_logreg)

        svm_model = SVC(probability=True)
        svm_model.fit(X_train, y_train)
        svm_predictions = svm_model.predict(X_test)
        self.visualize_svm_decision_boundary(X_train, y_train, svm_model, plot_filename_svm)

    def visualize_rf_predictions(self, X_test, y_test, predictions, plot_filename):
        plt.figure(figsize=(10, 5))
        plt.plot(y_test, label='Actual Category', color='red', marker='o')
        plt.plot(predictions, label='Predicted Category', color='green', marker='x')
        plt.title('Random Forest Predictions')
        plt.xlabel('Index')
        plt.ylabel('Category')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
        self.display_plot(plot_filename, 'Random Forest Predictions')
    def visualize_lr_distribution(self, predictions, plot_filename):
        plt.figure(figsize=(10, 5))
        plt.hist(predictions, bins=10, color='blue', alpha=0.7)
        plt.title('Distribution of Linear Regression Predictions')
        plt.xlabel('Predicted Category')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
        self.display_plot(plot_filename, 'Distribution of Linear Regression Predictions')

    def visualize_logreg_scatter(self, X_test, y_test, predictions, plot_filename):
        plt.figure(figsize=(10, 5))
        plt.scatter(range(len(y_test)), y_test, label='Actual Category', color='red',
                    alpha=0.5)
        plt.scatter(range(len(predictions)), predictions, label='Predicted Category', color='green',
                    alpha=0.5)
        plt.title('Logistic Regression Predictions')
        plt.xlabel('Index')
        plt.ylabel('Category')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
        self.display_plot(plot_filename, 'Logistic Regression Predictions')

    def visualize_svm_decision_boundary(self, X_train, y_train, model, plot_filename):
        plt.figure(figsize=(10, 5))

        x_min, x_max = X_train['WindSpeed_kmh'].min() - 1, X_train['WindSpeed_kmh'].max() + 1
        y_min, y_max = X_train['AtmosphericPressure_hPa'].min() - 1, X_train['AtmosphericPressure_hPa'].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        Z = model.predict(
            np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.ravel().shape)])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X_train['WindSpeed_kmh'], X_train['AtmosphericPressure_hPa'], c=y_train, edgecolors='k', marker='o')
        plt.title('SVM Decision Boundary')
        plt.xlabel('Wind Speed (km/h)')
        plt.ylabel('Atmospheric Pressure (hPa)')
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
        self.display_plot(plot_filename, 'SVM Decision Boundary')
    classify_hurricanes('hurricane/Hurricane_Data.csv', new_example=np.array([[100, 980, 75]]))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dashboard = Dashboard()
    dashboard.show()
    sys.exit(app.exec_())