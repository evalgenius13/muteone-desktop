import os, gc, subprocess, tempfile
import torch, torchaudio
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from demucs.pretrained import get_model
from demucs.apply import apply_model


class ModelManager:
    """Handles loading Demucs/MDX models and cleanup."""
    def __init__(self):
        self.failed_models = set()
        try:
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except Exception:
            has_mps = False
        if torch.cuda.is_available():
            self.device = "cuda"
        elif has_mps:
            self.device = "mps"
        else:
            self.device = "cpu"

    def get_models_for_task(self, task, high_quality=False):
        if task == "vocals" and high_quality:
            return ["UVR-MDX-NET-Voc_FT.onnx"]
        return ["htdemucs_ft"]

    def is_mdx_model(self, model_name):
        return "MDX" in model_name or "UVR" in model_name

    def load_model_safely(self, model_name, status_callback=None):
        if model_name in self.failed_models:
            raise Exception(f"Model {model_name} previously failed")
        try:
            if self.is_mdx_model(model_name):
                from audio_separator.separator import Separator
                if status_callback:
                    status_callback(f"Loading MDX {model_name}...")
                sep = Separator(
                    output_dir=os.path.join(os.path.dirname(__file__), "outputs"),
                    output_format="wav"
                )
                sep.load_model(model_name)
                return sep
            else:
                if status_callback:
                    status_callback(f"Loading Demucs {model_name}...")
                model = get_model(model_name)
                model.eval()
                return model
        except Exception as e:
            self.failed_models.add(model_name)
            raise Exception(f"Failed to load {model_name}: {e}")

    def cleanup_torch(self, *objs):
        for o in objs:
            try:
                del o
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()


class AudioProcessor(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    processing_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, input_file, output_dir, instruments_to_remove, high_quality=False):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.instruments_to_remove = instruments_to_remove
        self.high_quality = high_quality
        self.model_manager = ModelManager()

    def run(self):
        try:
            device = self.model_manager.device
            self.status_updated.emit("Loading audio...")
            self.progress_updated.emit(10)

            waveform, sample_rate = torchaudio.load(self.input_file)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2]
            waveform = waveform.to(torch.float32)

            instrument_to_remove = self.instruments_to_remove[0]
            model_name = "htdemucs_ft"

            self.status_updated.emit(f"Processing with {model_name}...")
            self.progress_updated.emit(40)

            model = self.model_manager.load_model_safely(model_name, self.status_updated.emit)
            model = model.to(device)

            with torch.inference_mode():
                sources = apply_model(model, waveform.unsqueeze(0), device=device)[0]

            self.progress_updated.emit(70)

            final_mix = torch.zeros_like(sources[0])
            for i, label in enumerate(model.sources):
                if label != instrument_to_remove:
                    final_mix += sources[i]

            self.model_manager.cleanup_torch(model, sources)
            self.save_output(final_mix, sample_rate)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def save_output(self, final_mix, sample_rate):
        import os
        from pathlib import Path
        input_path = Path(self.input_file)
        instrument_name = self.instruments_to_remove[0] if self.instruments_to_remove else "unknown"
        base_name = f"{input_path.stem}_muteone_no_{instrument_name}"
        output_path = os.path.join(self.output_dir, f"{base_name}.wav")

        counter = 0
        while os.path.exists(output_path):
            counter += 1
            output_path = os.path.join(self.output_dir, f"{base_name}_{counter}.wav")

        torchaudio.save(output_path, final_mix.detach().cpu(), sample_rate)
        self.progress_updated.emit(100)
        self.processing_complete.emit(output_path)
