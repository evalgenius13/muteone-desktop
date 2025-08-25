import sounddevice as sd
import soundfile as sf
import numpy as np


class AudioPlayer:
    """Playback with sounddevice, safe stop/pause handling"""

    def __init__(self):
        self.stream = None
        self.data = None
        self.samplerate = 44100
        self.position = 0
        self.is_playing = False
        self.output_device = None

    def load(self, file_path):
        self.data, self.samplerate = sf.read(file_path, dtype="float32")
        if self.data.ndim == 1:
            # ensure 2D shape (frames, channels)
            self.data = np.expand_dims(self.data, axis=1)
        self.position = 0

    def _callback(self, outdata, frames, time, status):
        if status:
            print("Playback:", status)
        if self.position >= len(self.data):
            outdata.fill(0)
            raise sd.CallbackStop()
        else:
            end = min(self.position + frames, len(self.data))
            chunk = self.data[self.position:end]
            if chunk.shape[0] < frames:
                pad = np.zeros((frames - chunk.shape[0], self.data.shape[1]), dtype="float32")
                chunk = np.vstack((chunk, pad))
            outdata[:] = chunk
            self.position = end

    def play(self):
        if self.data is None:
            return
        self.stop()  # stop existing before starting new
        try:
            dev_info = sd.query_devices(self.output_device, "output")
            channels = min(self.data.shape[1], dev_info["max_output_channels"])
            self.stream = sd.OutputStream(
                samplerate=self.samplerate,
                channels=channels,
                device=self.output_device,
                callback=self._callback,
                finished_callback=self._on_finished,
            )
            self.stream.start()
            self.is_playing = True
        except Exception as e:
            print(f"Playback failed: {e}")
            self.is_playing = False

    def _on_finished(self):
        self.is_playing = False
        self.position = 0

    def pause(self):
        if self.stream and self.stream.active:
            self.stream.stop()
            self.is_playing = False

    def stop(self):
        if self.stream:
            try:
                if self.stream.active:
                    self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.is_playing = False
        self.position = 0

    def seek(self, seconds):
        if self.data is None:
            return
        frame = int(seconds * self.samplerate)
        self.position = max(0, min(frame, len(self.data)))

    def get_position(self):
        return self.position / self.samplerate if self.data is not None else 0.0

    def get_duration(self):
        return len(self.data) / self.samplerate if self.data is not None else 0.0
