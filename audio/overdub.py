import sounddevice as sd
import soundfile as sf
import numpy as np


class OverdubRecorder:
    def __init__(self):
        self.backing_data = None
        self.samplerate = 44100
        self.input_device = None
        self.output_device = None

    def set_devices(self, input_device=None, output_device=None):
        self.input_device = input_device
        self.output_device = output_device

    def load_backing(self, filepath):
        self.backing_data, self.samplerate = sf.read(filepath, dtype="float32")
        if self.backing_data.ndim == 1:
            self.backing_data = np.expand_dims(self.backing_data, axis=1)

    def record_overdub(self, duration, out_path):
        if self.backing_data is None:
            raise RuntimeError("No backing loaded for overdub")

        # allocate buffer
        recorded = np.zeros((int(duration * self.samplerate), 1), dtype="float32")
        backing = self.backing_data[:len(recorded)]
        idx = 0

        def callback(indata, outdata, frames, time_info, status):
            nonlocal idx
            end = idx + frames
            if end > len(recorded):
                outdata.fill(0)
                raise sd.CallbackStop()
            recorded[idx:end, 0] = indata[:, 0]
            out_chunk = backing[idx:end] if idx < len(backing) else np.zeros((frames, backing.shape[1]))
            outdata[:len(out_chunk)] = out_chunk
            if len(out_chunk) < len(outdata):
                outdata[len(out_chunk):] = 0
            idx = end

        with sd.Stream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            device=(self.input_device, self.output_device),
            callback=callback
        ):
            sd.sleep(int(duration * 1000))

        # mix backing + recorded (pan recorded center)
        mixed = backing.copy()
        mixed[:len(recorded), 0] += recorded[:, 0]
        sf.write(out_path, mixed, self.samplerate)
        return out_path
