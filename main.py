#!/usr/bin/env python3
"""MuteOne Desktop App - Audio Source Separation Tool"""

import sys
import os
import ssl
import warnings
import subprocess
import time
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QLabel, QFileDialog, QProgressBar,
    QTextEdit, QGroupBox, QMessageBox, QRadioButton, QButtonGroup,
    QCheckBox
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

import torch
import torchaudio
import numpy as np

# Local imports
from audio.recording import LiveRecorder, LiveLevelMonitor
from audio.processor import AudioProcessor
from audio.player import AudioPlayer
from ui.level_meter import LevelMeter
from ui.waveform_widget import AudioEditorSection
from ui.recording_booth import RecordingBooth

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


class MuteOneApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_file = None
        self.output_dir = str(Path.home() / "Downloads")
        self.processing_thread = None
        self.recording_thread = None
        self.level_monitor = None
        self.is_recording = False
        self.recorded_file_path = None
        self.playback_process = None
        self.is_playing = False

        # Editor-specific playback
        self.audio_player = AudioPlayer()
        self.editor_is_playing = False
        self.playback_timer = None
        self.playback_position_sec = 0.0

        os.makedirs(self.output_dir, exist_ok=True)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("MuteOne - AI Audio Source Separation")
        self.setGeometry(100, 100, 700, 750)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Title
        title = QLabel("MuteOne")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Mute one instrument, keep the rest")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #666; margin-bottom: 20px;")
        layout.addWidget(subtitle)

        # === Recording Section ===
        recording_group = QGroupBox("Live Recording")
        recording_layout = QVBoxLayout(recording_group)

        buttons_layout = QHBoxLayout()
        self.start_recording_btn = QPushButton("🔴 Record")
        self.start_recording_btn.clicked.connect(self.start_recording)
        self.stop_recording_btn = QPushButton("⏹️ Stop")
        self.stop_recording_btn.clicked.connect(self.stop_recording)
        self.stop_recording_btn.setEnabled(False)
        buttons_layout.addWidget(self.start_recording_btn)
        buttons_layout.addWidget(self.stop_recording_btn)
        recording_layout.addLayout(buttons_layout)

        self.recording_status = QLabel("Ready to record")
        recording_layout.addWidget(self.recording_status)
        self.recording_timer = QLabel("00:00")
        self.recording_timer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        recording_layout.addWidget(self.recording_timer)

        # Level meters
        self.level_meter = LevelMeter(channels=2)
        recording_layout.addWidget(self.level_meter)

        layout.addWidget(recording_group)

        # File input
        file_group = QGroupBox("Select Audio File")
        file_layout = QHBoxLayout(file_group)
        self.select_file_btn = QPushButton("Choose File")
        self.select_file_btn.clicked.connect(self.select_file)
        file_layout.addWidget(self.select_file_btn)
        self.file_label = QLabel("No file selected")
        file_layout.addWidget(self.file_label, 1)

        # Booth button (greyed out until file/recording)
        self.open_booth_btn = QPushButton("Open Recording Booth")
        self.open_booth_btn.setEnabled(False)
        self.open_booth_btn.clicked.connect(self.open_recording_booth)
        file_layout.addWidget(self.open_booth_btn)

        layout.addWidget(file_group)

        # Audio Editor (collapsible)
        self.audio_editor = AudioEditorSection()
        self.audio_editor.play_requested.connect(self.editor_play_pause_audio)
        self.audio_editor.stop_requested.connect(self.editor_stop_audio)
        self.audio_editor.seek_requested.connect(self.editor_seek_audio)
        layout.addWidget(self.audio_editor)

        # Instrument selection
        instrument_group = QGroupBox("Mute One Instrument")
        instrument_layout = QVBoxLayout(instrument_group)
        self.instrument_group_buttons = QButtonGroup(self)
        self.instrument_group_buttons.setExclusive(True)
        self.instrument_radios = {}
        instruments = [
            ("vocals", "Vocals"),
            ("drums", "Drums"),
            ("bass", "Bass"),
            ("other", "Other"),
        ]
        for instrument_key, instrument_label in instruments:
            radio = QRadioButton(instrument_label)
            self.instrument_radios[instrument_key] = radio
            self.instrument_group_buttons.addButton(radio)
            instrument_layout.addWidget(radio)
        self.instrument_radios["vocals"].setChecked(True)

        self.high_quality_checkbox = QCheckBox("High Quality Vocal Removal")
        instrument_layout.addWidget(self.high_quality_checkbox)
        layout.addWidget(instrument_group)

        # Process
        self.process_btn = QPushButton("Process Audio")
        self.process_btn.clicked.connect(self.process_audio)
        self.process_btn.setEnabled(False)
        layout.addWidget(self.process_btn)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(140)
        layout.addWidget(self.status_text)

        # Start idle monitoring
        self.start_level_monitoring()

    # ===== Recording =====
    def start_level_monitoring(self):
        self.level_monitor = LiveLevelMonitor()
        self.level_monitor.audio_level_updated.connect(self.update_audio_levels)
        self.level_monitor.error_occurred.connect(self.on_monitoring_error)
        self.level_monitor.start_monitoring()

    def stop_level_monitoring(self):
        if self.level_monitor:
            self.level_monitor.stop_monitoring()
            self.level_monitor.wait(1000)

    def update_audio_levels(self, audio_data):
        self.level_meter.update_levels(audio_data)

    def on_monitoring_error(self, error_message):
        self.add_status(f"Monitor error: {error_message}")

    def start_recording(self):
        self.stop_level_monitoring()
        self.recording_thread = LiveRecorder()
        self.recording_thread.recording_started.connect(self.on_recording_started)
        self.recording_thread.recording_stopped.connect(self.on_recording_stopped)
        self.recording_thread.recording_time_updated.connect(
            lambda t: self.recording_timer.setText(t)
        )
        self.recording_thread.audio_level_updated.connect(self.update_audio_levels)
        self.recording_thread.start_recording()

    def stop_recording(self):
        if self.recording_thread:
            self.recording_thread.stop_recording()

    def on_recording_started(self):
        self.is_recording = True
        self.start_recording_btn.setEnabled(False)
        self.stop_recording_btn.setEnabled(True)
        self.recording_status.setText("Recording...")

    def on_recording_stopped(self, temp_file_path):
        self.is_recording = False
        self.start_recording_btn.setEnabled(True)
        self.stop_recording_btn.setEnabled(False)
        self.recording_status.setText("Recording complete")
        self.recorded_file_path = temp_file_path
        self.input_file = temp_file_path
        self.file_label.setText(os.path.basename(temp_file_path))
        self.process_btn.setEnabled(True)
        self.open_booth_btn.setEnabled(True)
        self.audio_editor.load_audio(temp_file_path)
        self.start_level_monitoring()

    # ===== Audio Editor =====
    def editor_play_pause_audio(self):
        if not self.input_file:
            return
        if self.editor_is_playing:
            self.audio_player.pause()
            self.editor_is_playing = False
            self.audio_editor.set_play_button_state(False)
        else:
            self.audio_player.load(self.input_file)
            self.audio_player.play()
            self.editor_is_playing = True
            self.audio_editor.set_play_button_state(True)
            self.playback_position_sec = 0.0
            self.playback_timer = QTimer(self)
            self.playback_timer.timeout.connect(self.update_editor_cursor)
            self.playback_timer.start(30)

    def update_editor_cursor(self):
        if not self.audio_player.is_playing:
            return
        pos = self.audio_player.get_position()
        dur = self.audio_player.get_duration()
        if pos >= dur:
            self.editor_stop_audio()
            return
        self.audio_editor.update_playback_position(pos)

    def editor_stop_audio(self):
        self.audio_player.stop()
        self.editor_is_playing = False
        if self.playback_timer:
            self.playback_timer.stop()
        self.audio_editor.update_playback_position(0.0)
        self.audio_editor.set_play_button_state(False)

    def editor_seek_audio(self, pos_seconds):
        self.audio_player.seek(pos_seconds)
        self.audio_editor.update_playback_position(pos_seconds)

    # ===== Booth =====
    def open_recording_booth(self):
        if not self.input_file or not os.path.exists(self.input_file):
            QMessageBox.warning(self, "Error", "No audio file loaded for the booth.")
            return
        try:
            booth = RecordingBooth(self.input_file, self.output_dir)
            booth.exec()
        except Exception as e:
            self.add_status(f"Booth error: {e}")

    # ===== File selection =====
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg)"
        )
        if file_path:
            self.input_file = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.process_btn.setEnabled(True)
            self.open_booth_btn.setEnabled(True)
            self.audio_editor.load_audio(file_path)

    # ===== Processing =====
    def process_audio(self):
        if not self.input_file:
            return
        instruments_to_remove = [
            key for key, radio in self.instrument_radios.items() if radio.isChecked()
        ]
        if not instruments_to_remove:
            QMessageBox.warning(self, "Error", "Select an instrument to mute")
            return

        high_quality = instruments_to_remove[0] == "vocals" and self.high_quality_checkbox.isChecked()
        self.processing_thread = AudioProcessor(
            self.input_file, self.output_dir, instruments_to_remove, high_quality
        )
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.status_updated.connect(self.add_status)
        self.processing_thread.processing_complete.connect(self.processing_finished)
        self.processing_thread.start()
        self.add_status(f"Processing {os.path.basename(self.input_file)}...")

    def update_progress(self, value):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(value)

    def processing_finished(self, output_path):
        self.add_status(f"Saved: {output_path}")
        QMessageBox.information(self, "Done", f"Saved file:\n{output_path}")

    def add_status(self, message):
        self.status_text.append(f"• {message}")
        self.status_text.verticalScrollBar().setValue(
            self.status_text.verticalScrollBar().maximum()
        )


def main():
    app = QApplication(sys.argv)
    window = MuteOneApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
