import os
import torchaudio
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QGroupBox
)
from PyQt6.QtCore import Qt
from audio.player import AudioPlayer
from ui.level_meter import LevelMeter


class RecordingBooth(QDialog):
    def __init__(self, input_file, output_dir):
        super().__init__()
        self.setWindowTitle("Recording Booth")
        self.setGeometry(200, 200, 600, 400)

        self.input_file = input_file
        self.output_dir = output_dir
        self.audio_player = AudioPlayer()

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Input device select
        input_group = QGroupBox("Mic Input")
        input_layout = QHBoxLayout(input_group)
        self.input_select = QComboBox()
        self.input_select.addItem("None")
        self.input_select.addItem("Built-in Mic")
        self.input_select.addItem("External Interface")
        input_layout.addWidget(QLabel("Input:"))
        input_layout.addWidget(self.input_select)
        layout.addWidget(input_group)

        # Mic Level meters
        self.meter = LevelMeter(channels=2)
        layout.addWidget(self.meter)

        # Buttons
        buttons_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.clicked.connect(self.play_audio)
        buttons_layout.addWidget(self.play_btn)

        self.pause_btn = QPushButton("⏸️ Pause")
        self.pause_btn.clicked.connect(self.pause_audio)
        buttons_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self.stop_audio)
        buttons_layout.addWidget(self.stop_btn)

        self.record_btn = QPushButton("🎙️ Record")
        self.record_btn.setEnabled(False)  # disabled until mic selected
        buttons_layout.addWidget(self.record_btn)

        layout.addLayout(buttons_layout)

        # Back button
        self.close_btn = QPushButton("⬅️ Back to Main")
        self.close_btn.clicked.connect(self.close)
        layout.addWidget(self.close_btn)

        # Enable record only if mic selected
        self.input_select.currentIndexChanged.connect(self.update_record_state)

    def update_record_state(self):
        if self.input_select.currentIndex() == 0:  # "None"
            self.record_btn.setEnabled(False)
        else:
            self.record_btn.setEnabled(True)

    # ===== Playback logic =====
    def play_audio(self):
        if not self.input_file or not os.path.exists(self.input_file):
            return
        self.audio_player.load(self.input_file)
        self.audio_player.play()

    def pause_audio(self):
        self.audio_player.pause()

    def stop_audio(self):
        self.audio_player.stop()
