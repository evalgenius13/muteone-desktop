from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtCore import Qt
import numpy as np


class LevelMeter(QWidget):
    def __init__(self, channels=2, segments=30):
        super().__init__()
        self.channels = channels
        self.levels_db = [-90.0] * channels  # peak in dBFS per channel
        self.segments = segments
        self.setMinimumHeight(80)
        self.setMaximumHeight(120)

    def update_levels(self, audio_bytes):
        """Update peak dBFS levels from raw audio data"""
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

        if self.channels == 2:
            samples = samples.reshape(-1, 2)
            peaks = np.max(np.abs(samples), axis=0) / 32768.0
        else:
            peaks = [np.max(np.abs(samples)) / 32768.0]

        # Convert to dBFS
        self.levels_db = [20 * np.log10(p + 1e-6) for p in peaks]
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w = self.width()
        h = self.height()
        row_h = h // self.channels
        seg_w = (w - 60) // self.segments  # leave space for labels

        for ch, db in enumerate(self.levels_db):
            y = ch * row_h

            # Normalize db range: -60 dBFS → 0 dBFS = full scale
            norm = (db + 60) / 60.0
            level_fraction = max(0.0, min(1.0, norm))
            filled_segments = int(level_fraction * self.segments)

            for i in range(self.segments):
                seg_x = 40 + i * seg_w
                seg_y = y + 5
                seg_h = row_h - 10

                # Zone coloring by position
                if i < int(self.segments * 0.6):
                    base_color = QColor(0, 200, 0)   # green
                elif i < int(self.segments * 0.85):
                    base_color = QColor(255, 200, 0) # yellow
                else:
                    base_color = QColor(255, 0, 0)   # red zone

                # Light up if within current level
                if i < filled_segments:
                    # ✅ Only show red if actual peak > -6 dBFS
                    if base_color == QColor(255, 0, 0) and db <= -6:
                        p.fillRect(int(seg_x), int(seg_y), int(seg_w - 2), int(seg_h), QColor(40, 40, 40))
                    else:
                        p.fillRect(int(seg_x), int(seg_y), int(seg_w - 2), int(seg_h), base_color)
                else:
                    # unlit segment
                    p.fillRect(int(seg_x), int(seg_y), int(seg_w - 2), int(seg_h), QColor(40, 40, 40))

            # Channel label (L / R)
            p.setPen(Qt.GlobalColor.white)
            label = "L" if ch == 0 else "R"
            p.drawText(10, y + row_h - 10, label)
