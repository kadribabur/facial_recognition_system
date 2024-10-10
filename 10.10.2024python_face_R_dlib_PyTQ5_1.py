# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:19:02 2024

@author: babur
"""

import sys
import os
import csv
import numpy as np
import cv2 as cv
import dlib
import threading
import locale
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt, QThread
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QMessageBox, QTabWidget, QTextEdit, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap
from colorama import init as colorama_init
from colorama import Fore, Back, Style
import Face_Recognition as f_r  # Varsayılan olarak face_recognition kütüphanesi kullanılmalı
import winsound  # Windows için ses bildirimleri
# from plyer import notification  # Alternatif olarak plyer kullanılabilir

# Initialize colorama
colorama_init()

# Locale ayarları
locale.setlocale(locale.LC_ALL, '')
current_locale, encoding = locale.getlocale()
language = 'en'  # Varsayılan dili İngilizce olarak ayarlayın

# Blink Detection Parameters
EAR_THRESHOLD = 0.2  # Göz kırpma için eşik değeri
CONSEC_FRAMES = 3

# Paths
PREDICTOR_PATH = r"C:\Users\babur\spiderPython\shape_predictor_68_face_landmarks (1).dat"
EYES_CASCADE_PATH = r"C:\Users\babur\Downloads\eye.xml"

# Verify paths
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(f"Predictor dosyası bulunamadı: {PREDICTOR_PATH}")
if not os.path.exists(EYES_CASCADE_PATH):
    raise FileNotFoundError(f"Göz cascade dosyası bulunamadı: {EYES_CASCADE_PATH}")

# Dlib setup
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Blink Counter
blink_count = 0
COUNTER = 0

# Recognition Count
recognition_count = {}

# People Data (This should ideally come from a database or a separate file)
people = [
    
    {"name": "Arda", "surname": "Güler", "age": 19, "img_path": "C:\\Users\\babur\\kadr1\\arda_guler2.jpg"},
    
    {"name": "Alexander", "surname": "Djiko", "age": 25, "img_path": "C:\\Users\\babur\\kadr1\\djiko.jpg"}
    
    # ... (liste devam ediyor)
]
# Initialize recognition_count
recognition_count = {person["name"]: 0 for person in people}

# Load known faces
known_face_encodings = []
known_face_metadata = []

for person in people:
    if not os.path.exists(person["img_path"]):
        print(f"{person['img_path']} bulunamadı. Lütfen dosya yolunu kontrol ediniz ...")
        continue
    image = f_r.load_image_file(person["img_path"])
    face_locations = f_r.face_locations(image)
    face_encodings = f_r.face_encodings(image, face_locations)
    if face_encodings:
        known_face_encodings.append(face_encodings[0])
        known_face_metadata.append(person)


def log_recognition(name, surname, age):
    with open("recognation_log.csv", "a", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), name, surname, age])


def notify(title, message):
    # Windows için winsound kullanımı
    winsound.Beep(2500, 1000)
    QMessageBox.warning(None, title, message)
    # Alternatif olarak plyer bildirimleri:
    # notification.notify(title=title, message=message, timeout=5)


def eye_aspect_ratio(eye_landmarks):
    # Göz kırpma oranını hesaplamak için göz noktalarını kullanın
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = False
        self.cap = None

    def run(self):
        global blink_count, COUNTER
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            self.update_log_signal.emit("Kamera açılamadı.")
            return

        eyes_cascade = cv.CascadeClassifier(EYES_CASCADE_PATH)
        if eyes_cascade.empty():
            self.update_log_signal.emit("Göz cascade dosyası yüklenemedi.")
            return

        while self._run_flag:
            ret, frame = self.cap.read()
            if not ret:
                self.update_log_signal.emit("Kamera okuma hatası.")
                break

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame_face_locations = f_r.face_locations(frame_rgb)
            frame_face_encodings = f_r.face_encodings(frame_rgb, frame_face_locations)

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            for (top, right, bottom, left), frame_face_encoding in zip(frame_face_locations, frame_face_encodings):
                matches = f_r.compare_faces(known_face_encodings, frame_face_encoding)
                name = "Unknown :)"
                surname = ""
                age = ""

                if True in matches:
                    first_match_index = matches.index(True)
                    matched_person = known_face_metadata[first_match_index]
                    name = matched_person["name"]
                    surname = matched_person["surname"]
                    age = matched_person["age"]
                    recognition_count[name] += 1  # Tanıma sayısını artır
                    log_recognition(name, surname, age)

                else:
                    # Bilinmeyen yüz algılandığında bildirim
                    notify("Uyarı", "Bilinmeyen bir yüz algılandı!")

                # Yüzü çerçevele ve ismi, soyismi ve yaşı yaz
                cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv.putText(frame, f"{name} {surname}, {age} (Tespit: {recognition_count.get(name,0)} kez)",
                           (left, bottom + 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 0, 100), 2)

                # Gözleri algıla ve göz kırpma tespiti yap
                faces = detector(gray, 0)
                for face in faces:
                    shape = predictor(gray, face)
                    landmarks = np.array([[p.x, p.y] for p in shape.parts()])

                    # Göz bölgeleri (sol ve sağ)
                    left_eye = landmarks[36:42]
                    right_eye = landmarks[42:48]

                    left_EAR = eye_aspect_ratio(left_eye)
                    right_EAR = eye_aspect_ratio(right_eye)
                    ear = (left_EAR + right_EAR) / 2.0

                    if ear < EAR_THRESHOLD:
                        COUNTER += 1
                    else:
                        if COUNTER >= CONSEC_FRAMES:
                            blink_count += 1
                            cv.putText(frame, "Göz Kırpıldı!", (left, bottom + 80),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        COUNTER = 0

                    cv.polylines(frame, [left_eye], True, (255, 0, 0), 1)
                    cv.polylines(frame, [right_eye], True, (255, 0, 0), 1)

                # Mesafe hesapla
                face_height = bottom - top
                if face_height != 0:
                    distance = 500 / face_height  # Yüz yüksekliğine göre basit mesafe tahmini
                    cv.putText(frame, f"Mesafe Ölçümü: {distance:.2f} cm", (
                        left, bottom + 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)

                # Blink sayısını yazdır
                cv.putText(frame, f"Toplam Blink: {blink_count}", (left, bottom + 100),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            self.change_pixmap_signal.emit(frame)

        # Release the video capture when the thread is stopped
        self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Yüz Tanıma ve Göz Kırpma Tespiti")
        self.setGeometry(100, 100, 800, 600)

        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Tab Widget
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Main Tab
        self.tab_main = QWidget()
        self.tabs.addTab(self.tab_main, "Ana Sayfa")
        self.tab_main_layout = QVBoxLayout()
        self.tab_main.setLayout(self.tab_main_layout)

        # Video Display
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("background-color: black;")
        self.tab_main_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # Buttons Layout
        self.buttons_layout = QHBoxLayout()
        self.tab_main_layout.addLayout(self.buttons_layout)

        # Start Camera Button
        self.start_btn = QPushButton("Kamerayı Başlat")
        self.start_btn.clicked.connect(self.start_camera)
        self.buttons_layout.addWidget(self.start_btn)

        # Stop Camera Button
        self.stop_btn = QPushButton("Kamerayı Durdur")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)
        self.buttons_layout.addWidget(self.stop_btn)

        # Register User Button
        self.register_btn = QPushButton("Kullanıcı Kayıt")
        self.register_btn.clicked.connect(self.register_user)
        self.buttons_layout.addWidget(self.register_btn)

        # Settings Button
        self.settings_btn = QPushButton("Ayarlar")
        self.settings_btn.clicked.connect(self.open_settings)
        self.buttons_layout.addWidget(self.settings_btn)

        # Statistics Button
        self.stats_btn = QPushButton("İstatistikleri Göster")
        self.stats_btn.clicked.connect(self.show_statistics)
        self.buttons_layout.addWidget(self.stats_btn)

        # Logs Tab
        self.tab_logs = QWidget()
        self.tabs.addTab(self.tab_logs, "Tanıma Logları")
        self.tab_logs_layout = QVBoxLayout()
        self.tab_logs.setLayout(self.tab_logs_layout)

        # Logs Text Edit
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.tab_logs_layout.addWidget(self.log_text)

        # Initialize Video Thread
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_log_signal.connect(self.update_log)
        self.thread.start()

        # Initial Log Update
        self.update_logs()

    def start_camera(self):
        self.thread._run_flag = True
        self.thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.update_log("Kamera başlatıldı.")

    def stop_camera(self):
        self.thread.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.update_log("Kamera durduruldu.")

    def update_image(self, cv_img):
        """Convert from an OpenCV image to QPixmap and display it."""
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(QPixmap.fromImage(p))

    def register_user(self):
        # Kullanıcı bilgilerini almak için diyaloglar
        name, ok = QtWidgets.QInputDialog.getText(self, "Kullanıcı Kayıt", "İsim:")
        if not ok or not name:
            return

        surname, ok = QtWidgets.QInputDialog.getText(self, "Kullanıcı Kayıt", "Soyisim:")
        if not ok or not surname:
            return

        age, ok = QtWidgets.QInputDialog.getInt(self, "Kullanıcı Kayıt", "Yaş:")
        if not ok:
            return

        # Fotoğraf seçme
        options = QFileDialog.Options()
        img_path, _ = QFileDialog.getOpenFileName(self, "Fotoğraf Seç", "",
                                                  "Image Files (*.png *.jpg *.jpeg)", options=options)
        if not img_path:
            QMessageBox.warning(self, "Uyarı", "Fotoğraf seçilmedi.")
            return

        # Yeni kişiyi listeye ekle
        new_person = {
            "name": name,
            "surname": surname,
            "age": age,
            "img_path": img_path
        }
        people.append(new_person)
        recognition_count[name] = 0

        # Yüz kodlamalarını güncelle
        image = f_r.load_image_file(img_path)
        face_locations = f_r.face_locations(image)
        face_encodings = f_r.face_encodings(image, face_locations)
        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_metadata.append(new_person)
            log_recognition(name, surname, age)
            QMessageBox.information(self, "Başarılı", f"{name} {surname} başarıyla kayıt edildi.")
        else:
            QMessageBox.warning(self, "Hata", "Yüz algılanamadı. Lütfen daha net bir fotoğraf seçin.")

    def open_settings(self):
        # Basit ayarlar penceresi
        settings_dialog = SettingsDialog(self)
        settings_dialog.exec_()

    def show_statistics(self):
        # İstatistikleri gösteren mesaj kutusu
        stats = "\n".join([f"{name}: {count} kez" for name, count in recognition_count.items()])
        QMessageBox.information(self, "Tanıma İstatistikleri", stats)

    def update_log(self, message):
        self.log_text.append(message)

    def update_logs(self):
        if os.path.exists("recognation_log.csv"):
            with open("recognation_log.csv", "r", encoding='utf-8') as file:
                log_content = file.read()
                self.log_text.setText(log_content)
        # Logları periyodik olarak güncelle
        QtCore.QTimer.singleShot(5000, self.update_logs)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__(parent)
        self.setWindowTitle("Ayarlar")
        self.setGeometry(150, 150, 300, 150)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Göz Kırpma Eşiği
        self.ear_label = QLabel("Göz Kırpma Eşiği:")
        self.layout.addWidget(self.ear_label)

        self.ear_input = QtWidgets.QLineEdit(str(EAR_THRESHOLD))
        self.layout.addWidget(self.ear_input)

        # Mesafe Eşiği
        self.distance_label = QLabel("Mesafe Eşiği (cm):")
        self.layout.addWidget(self.distance_label)

        self.distance_input = QtWidgets.QLineEdit("50")
        self.layout.addWidget(self.distance_input)

        # Kaydet Butonu
        self.save_btn = QPushButton("Kaydet")
        self.save_btn.clicked.connect(self.save_settings)
        self.layout.addWidget(self.save_btn)

    def save_settings(self):
        global EAR_THRESHOLD
        try:
            EAR_THRESHOLD = float(self.ear_input.text())
            distance_threshold = float(self.distance_input.text())
            QMessageBox.information(self, "Başarılı", "Ayarlar kaydedildi.")
            self.close()
        except ValueError:
            QMessageBox.warning(self, "Hata", "Geçerli bir sayı giriniz.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
