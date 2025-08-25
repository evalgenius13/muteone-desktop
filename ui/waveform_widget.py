import os
import numpy as np
import torch, torchaudio
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QFont


class WaveformWidget(QWidget):
    """Waveform display with trim + playback cursor"""
    seek_requested = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.waveform_data = None
        self.sample_rate = 44100
        self.duration = 0.0
        self.playback_position = 0.0

        # Trim markers
        self.trim_start = None
        self.trim_end = None
        self.dragging_marker = None

        self.bg_color = QColor(30, 30, 35)
        self.waveform_color = QColor(100, 200, 255)
        self.trim_color = QColor(255, 165, 0, 150)  # orange translucent
        self.cursor_color = QColor(255, 255, 255)

        self.setMinimumHeight(150)

    def load_audio(self, file_path):
        try:
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            self.sample_rate = sr
            self.duration = waveform.shape[1] / sr

            downsample_factor = max(1, waveform.shape[1] // 4000)
            self.waveform_data = waveform[0][::downsample_factor].numpy()

            self.trim_start = 0.0
            self.trim_end = self.duration
            self.update()
            return True
        except Exception as e:
            print(f"Error loading audio: {e}")
            return False

    def set_playback_position(self, position_seconds):
        self.playback_position = max(0.0, min(self.duration, position_seconds))
        self.update()

    def mousePressEvent(self, event):
        if self.waveform_data is None: return
        x = event.position().x()
        time_clicked = self.x_to_time(x)

        # Check if clicked near trim markers
        if abs(time_clicked - self.trim_start) < 0.5:
            self.dragging_marker = "start"
        elif abs(time_clicked - self.trim_end) < 0.5:
            self.dragging_marker = "end"
        else:
            self.seek_requested.emit(time_clicked)

    def mouseMoveEvent(self, event):
        if self.dragging_marker:
            time_pos = self.x_to_time(event.position().x())
            if self.dragging_marker == "start":
                self.trim_start = max(0.0, min(time_pos, self.trim_end - 0.1))
            elif self.dragging_marker == "end":
                self.trim_end = min(self.duration, max(time_pos, self.trim_start + 0.1))
            self.update()

    def mouseReleaseEvent(self, event):
        self.dragging_marker = None

    def x_to_time(self, x):
        widget_width = self.width() - 80
        return (x - 40) / widget_width * self.duration

    def time_to_x(self, t):
        widget_width = self.width() - 80
        return 40 + (t / self.duration * widget_width)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), self.bg_color)

        if self.waveform_data is None:
            p.setPen(QColor(200, 200, 200))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No audio loaded")
            return

        w, h = self.width(), self.height()
        center_y = h // 2
        draw_width = w - 80
        scale = (h // 2) * 0.8

        step = max(1, len(self.waveform_data) // draw_width)
        for i in range(0, len(self.waveform_data), step):
            x = 40 + (i / len(self.waveform_data)) * draw_width
            y = int(center_y - self.waveform_data[i] * scale)
            p.setPen(self.waveform_color)
            p.drawLine(int(x), center_y, int(x), y)

        # Trim region shading
        if self.trim_start is not None and self.trim_end is not None:
            x1, x2 = self.time_to_x(self.trim_start), self.time_to_x(self.trim_end)
            p.fillRect(int(x1), 0, int(x2 - x1), h, self.trim_color)

        # Playback cursor
        cursor_x = self.time_to_x(self.playback_position)
        p.setPen(QPen(self.cursor_color, 2))
        p.drawLine(int(cursor_x), 0, int(cursor_x), h)

        # Trim markers (flags)
        p.setPen(QPen(Qt.GlobalColor.red, 2))
        if self.trim_start is not None:
            x1 = self.time_to_x(self.trim_start)
            p.drawLine(int(x1), 0, int(x1), h)
        if self.trim_end is not None:
            x2 = self.time_to_x(self.trim_end)
            p.drawLine(int(x2), 0, int(x2), h)


class AudioEditorSection(QGroupBox):
    """Waveform editor with trim + play/pause toggle button"""
    play_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    seek_requested = pyqtSignal(float)

    def __init__(self):
        super().__init__("Audio Editor")
        self.audio_file = None
        self.is_playing = False
        self.waveform = WaveformWidget()
        self.setup_ui()
        self.hide()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(self.waveform)

        controls_layout = QHBoxLayout()

        # Single Play/Pause toggle
        self.play_pause_btn = QPushButton("▶️ Play")
        self.play_pause_btn.clicked.connect(self.play_requested.emit)
        controls_layout.addWidget(self.play_pause_btn)

        # Trim preview
        self.trim_btn = QPushButton("✂️ Trim Preview")
        self.trim_btn.clicked.connect(lambda: self.seek_requested.emit(self.waveform.trim_start or 0.0))
        controls_layout.addWidget(self.trim_btn)

        # Info label
        self.audio_info_label = QLabel("No audio loaded")
        controls_layout.addWidget(self.audio_info_label, 1)
        layout.addLayout(controls_layout)

    def load_audio(self, file_path):
        self.audio_file = file_path
        if self.waveform.load_audio(file_path):
            self.audio_info_label.setText(os.path.basename(file_path))
            self.show()
            return True
        return False

    def update_playback_position(self, pos_sec):
        self.waveform.set_playback_position(pos_sec)

    def set_play_button_state(self, is_playing: bool):
        """Update the play button label based on state"""
        if is_playing:
            self.play_pause_btn.setText("⏸️ Pause")
        else:
            self.play_pause_btn.setText("▶️ Play")
