import os
import time
import tempfile
import numpy as np
import torch
import torchaudio
import pyaudio

from PyQt6.QtCore import QThread, pyqtSignal


class LiveLevelMonitor(QThread):
    """Monitors audio levels without recording"""
    audio_level_updated = pyqtSignal(bytes)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_monitoring = False
        self.sample_rate = 44100
        self.channels = 2
        self.chunk_size = 1024
        self.format = pyaudio.paInt16

    def start_monitoring(self):
        self.is_monitoring = True
        self.start()

    def stop_monitoring(self):
        self.is_monitoring = False

    def run(self):
        try:
            p = pyaudio.PyAudio()
            input_device_index = None
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                name = info['name'].lower()
                if any(k in name for k in ['stereo mix', 'loopback', 'blackhole']):
                    if info['maxInputChannels'] >= 2:
                        input_device_index = i
                        break
            if input_device_index is None:
                input_device_index = p.get_default_input_device_info()['index']

            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=self.chunk_size
            )

            while self.is_monitoring:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_level_updated.emit(data)

            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            self.error_occurred.emit(f"Failed to start monitoring: {e}")


class LiveRecorder(QThread):
    """Records system audio to a temporary file"""
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal(str)
    recording_time_updated = pyqtSignal(str)
    audio_level_updated = pyqtSignal(bytes)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.audio_data = []
        self.temp_file_path = None
        self.start_time = 0
        self.sample_rate = 44100
        self.channels = 2
        self.chunk_size = 1024
        self.format = pyaudio.paInt16

    def start_recording(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.audio_data = []
        self.start_time = time.time()
        temp_dir = tempfile.gettempdir()
        self.temp_file_path = os.path.join(
            temp_dir, f"muteone_recording_{int(time.time())}.wav"
        )
        self.start()

    def stop_recording(self):
        self.is_recording = False

    def run(self):
        try:
            p = pyaudio.PyAudio()
            input_device_index = None
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                name = info['name'].lower()
                if any(k in name for k in ['stereo mix', 'loopback', 'blackhole']):
                    if info['maxInputChannels'] >= 2:
                        input_device_index = i
                        break
            if input_device_index is None:
                input_device_index = p.get_default_input_device_info()['index']

            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=self.chunk_size
            )

            self.recording_started.emit()

            while self.is_recording:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_data.append(data)
                self.audio_level_updated.emit(data)

                elapsed = time.time() - self.start_time
                minutes, seconds = divmod(int(elapsed), 60)
                self.recording_time_updated.emit(f"{minutes:02d}:{seconds:02d}")

            stream.stop_stream()
            stream.close()
            p.terminate()

            if self.audio_data and self.temp_file_path:
                self.save_recording()
        except Exception as e:
            self.error_occurred.emit(f"Recording failed: {e}")

    def save_recording(self):
        try:
            audio_bytes = b''.join(self.audio_data)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            if self.channels == 2:
                audio_np = audio_np.reshape(-1, 2)
            audio_float = audio_np.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(
                audio_float.T if self.channels == 2 else audio_float
            ).float()
            torchaudio.save(self.temp_file_path, audio_tensor, self.sample_rate)
            self.recording_stopped.emit(self.temp_file_path)
        except Exception as e:
            self.error_occurred.emit(f"Failed to save: {e}")


# --------------------------------------------------
# Shim class so main.py can still do "from ui.record import Recorder"
# --------------------------------------------------
class Recorder:
    """Simple wrapper for quick fixed-duration recordings (used by main.py)."""
    def __init__(self, samplerate=44100, channels=2):
        self.samplerate = samplerate
        self.channels = channels

    def record_to_file(self, duration=10):
        """
        Quick blocking recording — unlike LiveRecorder (threaded),
        this just captures N seconds and saves straight to ~/Downloads.
        """
        import sounddevice as sd, soundfile as sf
        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        os.makedirs(downloads, exist_ok=True)
        out_path = os.path.join(downloads, "omotiv_take.wav")

        print(f"[Recorder] Recording {duration}s → {out_path}")
        data = sd.rec(
            int(duration * self.samplerate),
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32"
        )
        sd.wait()
        sf.write(out_path, data, self.samplerate)
        print(f"[Recorder] Saved {out_path}")
        return out_path
