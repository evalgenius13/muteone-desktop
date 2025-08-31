import sys, os, threading, tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QRadioButton, QButtonGroup,
    QProgressBar, QMessageBox, QFileDialog, QComboBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPainter, QLinearGradient, QColor, QPen

from processor import AudioProcessor
from model_manager import ModelManager


# -------------------------
# Worker signals
# -------------------------
class WorkerSignals(QObject):
    status = pyqtSignal(str, str)   # msg, color
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)


# -------------------------
# Smooth custom level meter
# -------------------------
class SmoothLevelMeter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.level = 0
        self.setFixedHeight(12)
        self.setMinimumWidth(150)

    def set_level(self, level):
        self.level = max(0, min(100, level))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 10, 10))
        painter.setPen(QPen(QColor(30, 30, 30), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        if self.level > 0:
            gradient = QLinearGradient(0, 0, self.width(), 0)
            gradient.setColorAt(0.0, QColor(76, 175, 80))
            gradient.setColorAt(0.6, QColor(139, 195, 74))
            gradient.setColorAt(0.8, QColor(255, 235, 59))
            gradient.setColorAt(0.95, QColor(255, 152, 0))
            gradient.setColorAt(1.0, QColor(244, 67, 54))
            level_width = int((self.width() - 4) * self.level / 100)
            painter.fillRect(2, 2, level_width, self.height() - 4, gradient)


# -------------------------
# Main Window
# -------------------------
class MuteOne(QMainWindow):
    def __init__(self, processor: AudioProcessor):
        super().__init__()
        self.setWindowTitle("MuteOne by Omotiv Audio")
        self.setGeometry(200, 200, 720, 480)

        self.processor = processor
        self.current_file = None
        self.output_file = None
        self.last_muted = None
        self.playing = False
        self.signals = WorkerSignals()

        # signals
        self.signals.status.connect(self.update_status)
        self.signals.progress.connect(self.update_progress)
        self.signals.finished.connect(self.on_finished)
        self.signals.error.connect(self.show_error)

        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #e0e0e0;
                font-family: Arial;
                font-size: 13px;
            }
            QPushButton {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border-radius: 4px;
                padding: 5px 12px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
            QPushButton#primary {
                background-color: #1db954;
                color: white;
                font-weight: bold;
            }
            QPushButton#primary:hover {
                background-color: #1ed760;
            }
            QLabel#status {
                min-height: 26px;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
                qproperty-alignment: AlignCenter;
                font-size: 12px;
            }
        """)

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Upload
        self.upload_btn = QPushButton("Upload File")
        self.upload_btn.clicked.connect(self.upload_file)
        layout.addWidget(self.upload_btn)

        # Separation choices
        row = QHBoxLayout()
        row.addWidget(QLabel("Mute:"))
        self.radio_group = QButtonGroup()
        for r in ["Vocals", "Drums", "Bass", "Other"]:
            rb = QRadioButton(r)
            if r == "Vocals":
                rb.setChecked(True)
            self.radio_group.addButton(rb)
            row.addWidget(rb)
        layout.addLayout(row)

        # Original/Separated selection
        self.file_group = QButtonGroup()
        self.file_group.setExclusive(True)

        self.original_rb = QRadioButton("Original: (none)")
        self.separated_rb = QRadioButton("Separated: (none)")
        self.file_group.addButton(self.original_rb)
        self.file_group.addButton(self.separated_rb)
        self.original_rb.setChecked(True)
        layout.addWidget(self.original_rb)
        layout.addWidget(self.separated_rb)

        # Transport + level meter
        row = QHBoxLayout()
        self.rewind_btn = QPushButton("⏮")
        self.stop_btn = QPushButton("■")
        self.play_btn = QPushButton("▶")
        for btn in (self.rewind_btn, self.stop_btn, self.play_btn):
            btn.setFixedSize(40, 28)
            row.addWidget(btn)

        self.level_meter = SmoothLevelMeter()
        row.addWidget(self.level_meter)
        layout.addLayout(row)

        self.rewind_btn.clicked.connect(self.rewind_audio)
        self.stop_btn.clicked.connect(self.stop_audio)
        self.play_btn.clicked.connect(self.toggle_play)

        # Start/Cancel
        row = QHBoxLayout()
        self.start_btn = QPushButton("Start Separation")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start_separation)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_separation)
        row.addWidget(self.start_btn)
        row.addWidget(self.cancel_btn)
        layout.addLayout(row)

        # Progress + status
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("status")
        self.status_label.setFixedHeight(28)
        layout.addWidget(self.status_label)

        # Export
        row = QHBoxLayout()
        row.addStretch()
        self.format_combo = QComboBox()
        self.format_combo.addItems(["MP3", "WAV"])
        self.format_combo.setFixedWidth(80)
        self.export_btn = QPushButton("Export")
        self.export_btn.setObjectName("primary")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_file)
        row.addWidget(QLabel("Format:"))
        row.addWidget(self.format_combo)
        row.addWidget(self.export_btn)
        layout.addLayout(row)

    # -------------------------
    # File Handling
    # -------------------------
    def upload_file(self):
        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        file, _ = QFileDialog.getOpenFileName(self, "Select audio", downloads, "Audio Files (*.wav *.mp3)")
        if file:
            self.current_file = file
            self.original_rb.setText(f"Original: {os.path.basename(file)}")
            self.update_status(f"Loaded: {os.path.basename(file)}", "gray")

    # -------------------------
    # Transport + Level Meter
    # -------------------------
    def get_selected_file(self):
        if self.original_rb.isChecked():
            return self.current_file
        elif self.separated_rb.isChecked():
            return self.output_file
        return None

    def toggle_play(self):
        target = self.get_selected_file()
        if not target or not os.path.exists(target):
            self.show_error("No valid file selected for playback")
            return

        if self.playing:
            sd.stop()
            self.playing = False
            self.play_btn.setText("▶")
            self.update_status("Paused", "gray")
            if hasattr(self, "meter_timer"):
                self.meter_timer.stop()
            self.level_meter.set_level(0)
            return

        try:
            data, sr = sf.read(target, dtype="float32")
            sd.default.samplerate = sr

            def playback_fn():
                try:
                    with sd.OutputStream(samplerate=sr,
                                         channels=data.shape[1] if data.ndim > 1 else 1) as stream:
                        stream.write(data)
                finally:
                    self.playing = False
                    self.play_btn.setText("▶")
                    self.level_meter.set_level(0)

            threading.Thread(target=playback_fn, daemon=True).start()

            blocksize = int(sr * 0.05)
            envelope = []
            for start in range(0, len(data), blocksize):
                block = data[start:start+blocksize]
                if block.ndim > 1:
                    block = block.mean(axis=1)
                level = int(np.linalg.norm(block) * 10)
                envelope.append(min(100, level))

            self.meter_index = 0
            self.meter_data = envelope

            if hasattr(self, "meter_timer"):
                self.meter_timer.stop()
            self.meter_timer = QTimer()
            self.meter_timer.timeout.connect(self.update_meter_from_envelope)
            self.meter_timer.start(50)

            self.playing = True
            self.play_btn.setText("⏸")
            self.update_status(f"Playing: {os.path.basename(target)}", "green")

        except Exception as e:
            self.show_error(f"Playback failed: {e}")

    def update_meter_from_envelope(self):
        if not self.playing or self.meter_index >= len(self.meter_data):
            self.level_meter.set_level(0)
            if hasattr(self, "meter_timer"):
                self.meter_timer.stop()
            self.playing = False
            self.play_btn.setText("▶")
            return
        self.level_meter.set_level(self.meter_data[self.meter_index])
        self.meter_index += 1

    def stop_audio(self):
        sd.stop()
        self.playing = False
        self.play_btn.setText("▶")
        self.level_meter.set_level(0)
        self.update_status("Stopped", "gray")
        if hasattr(self, "meter_timer"):
            self.meter_timer.stop()

    def rewind_audio(self):
        self.stop_audio()
        self.toggle_play()

    # -------------------------
    # Separation
    # -------------------------
    def start_separation(self):
        if not self.current_file:
            self.show_error("No file uploaded")
            return

        checked = self.radio_group.checkedButton()
        if not checked:
            self.show_error("Select a stem to mute")
            return
        choice = checked.text().lower()
        self.last_muted = choice
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.start_btn.setEnabled(False)

        def worker():
            try:
                out = self.processor.run(
                    self.current_file, "", [choice],
                    progress_callback=self.signals.progress.emit,
                    status_callback=lambda msg: self.signals.status.emit(msg, "gray"),
                    cancelled=lambda: False
                )
                if out:
                    self.signals.finished.emit(out)
                else:
                    self.signals.status.emit("Failed", "red")
            except Exception as e:
                self.signals.error.emit(str(e))

        threading.Thread(target=worker, daemon=True).start()
        self.update_status(f"Processing {choice}...", "gray")

    def cancel_separation(self):
        self.processor.cancel()
        self.start_btn.setEnabled(True)
        self.update_status("Canceled", "red")
        self.progress.setVisible(False)

    def on_finished(self, filepath):
        self.progress.setVisible(False)
        self.start_btn.setEnabled(True)
        if filepath and os.path.exists(filepath):
            self.output_file = filepath
            self.separated_rb.setText(f"Separated: {os.path.basename(filepath)}")
            self.separated_rb.setChecked(True)
            self.export_btn.setEnabled(True)
            self.update_status("Success", "green")
        else:
            self.update_status("Error", "red")

    # -------------------------
    # Export
    # -------------------------
    def export_file(self):
        if not self.output_file or not os.path.exists(self.output_file):
            self.show_error("No processed file available to export")
            return

        fmt = self.format_combo.currentText().lower()
        base = os.path.splitext(os.path.basename(self.output_file))[0]
        suggested_name = f"{base}.{fmt}"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed File",
            os.path.join(os.path.expanduser("~"), "Documents", "MuteOne", suggested_name),
            "Audio Files (*.wav *.mp3)"
        )
        if not file_path:
            return

        try:
            data, sr = sf.read(self.output_file, dtype="float32")
            if fmt == "wav":
                sf.write(file_path, data, sr, format="WAV")
            elif fmt == "mp3":
                from pydub import AudioSegment
                temp_wav = os.path.join(tempfile.gettempdir(), f"{base}_temp.wav")
                sf.write(temp_wav, data, sr, format="WAV")
                AudioSegment.from_wav(temp_wav).export(
                    file_path,
                    format="mp3",
                    bitrate="320k",
                    parameters=["-ar", str(sr)]
                )
                os.remove(temp_wav)
            self.update_status(f"Exported to {file_path}", "green")
        except Exception as e:
            self.show_error(f"Export failed: {e}")

    # -------------------------
    # Status + Error
    # -------------------------
    def update_status(self, msg, color="gray"):
        colors = {
            "green": "background-color: #1db954; color: white;",
            "red": "background-color: #e22134; color: white;",
            "gray": "background-color: #333; color: #e0e0e0;"
        }
        style = colors.get(color, "background-color: #333; color: white;")
        self.status_label.setText(msg)
        self.status_label.setStyleSheet(
            style + "padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;"
        )

    def update_progress(self, val):
        self.progress.setValue(val)

    def show_error(self, msg):
        self.update_status("Error", "red")
        QMessageBox.critical(self, "Error", msg)


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    model_manager = ModelManager()
    processor = AudioProcessor(model_manager)
    app = QApplication(sys.argv)
    win = MuteOne(processor)
    win.show()
    sys.exit(app.exec())
