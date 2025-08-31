import os, torch, torchaudio, traceback, shutil, tempfile
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from demucs.apply import apply_model

# -------------------------
# Helpers
# -------------------------
def get_downloads_folder():
    return os.path.join(os.path.expanduser("~"), "Downloads")

def get_best_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _fix_tensor(tensor: torch.Tensor, batch: bool = False, device=None):
    """Ensure correct tensor shape + device placement for torchaudio/Demucs."""
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.from_numpy(tensor).float()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 3 and not batch:
        tensor = tensor.squeeze(0)
    tensor = tensor.float()
    if batch and tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

# -------------------------
# MDX Wrapper
# -------------------------
class MDXModelWrapper:
    """Wrapper for MDX vocal removal model (CPU/MPS/CUDA)."""
    def __init__(self, model_path, status_callback=None):
        self.model_path = model_path
        self.separator = None
        self.status_callback = status_callback

    def __call__(self, waveform, sample_rate=44100):
        try:
            if self.separator is None:
                if self.status_callback:
                    self.status_callback("Loading vocal removal engine...")
                from audio_separator.separator import Separator
                output_dir = os.path.join(tempfile.gettempdir(), "omotiv_outputs")
                os.makedirs(output_dir, exist_ok=True)
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                self.separator = Separator(output_dir=output_dir, output_format="wav")
                self.separator.load_model(self.model_path)

            if self.status_callback:
                self.status_callback("Separating vocals...")

            waveform_tensor = _fix_tensor(waveform)
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, "input_audio.wav")
            torchaudio.save(temp_path, waveform_tensor, sample_rate)

            vocals, instrumental = None, None
            try:
                output_files = self.separator.separate(temp_path)

                # Handle dict or list output
                if isinstance(output_files, dict):
                    for key, fp in output_files.items():
                        if os.path.exists(fp):
                            if "vocal" in key.lower():
                                vocals, _ = torchaudio.load(fp)
                            elif "instrumental" in key.lower() or "music" in key.lower():
                                instrumental, _ = torchaudio.load(fp)
                elif isinstance(output_files, list):
                    for fp in output_files:
                        if os.path.exists(fp):
                            fn = os.path.basename(fp).lower()
                            if "vocal" in fn or "voice" in fn:
                                vocals, _ = torchaudio.load(fp)
                            elif any(tag in fn for tag in ["instrumental", "no_vocals", "music"]):
                                instrumental, _ = torchaudio.load(fp)

                # Fallback scan
                if vocals is None or instrumental is None:
                    outdir = self.separator.output_dir
                    for fn in os.listdir(outdir):
                        if fn.endswith(".wav"):
                            fp = os.path.join(outdir, fn)
                            low = fn.lower()
                            if "vocal" in low and vocals is None:
                                vocals, _ = torchaudio.load(fp)
                            elif any(tag in low for tag in ["instrumental", "music", "no_vocals"]) and instrumental is None:
                                instrumental, _ = torchaudio.load(fp)

                if vocals is None or instrumental is None:
                    raise ValueError("Output missing vocal/instrumental stems")

                return [_fix_tensor(vocals), _fix_tensor(instrumental)]

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Vocal removal error: {str(e)}")
            raise

    def cleanup(self):
        if self.separator and hasattr(self.separator, "cleanup"):
            try:
                self.separator.cleanup()
            except Exception:
                pass
        self.separator = None

# -------------------------
# Audio Processor
# -------------------------
class AudioProcessor(QObject):
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    processing_finished = pyqtSignal(str)

    def __init__(self, model_manager, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.model_manager.device = get_best_device()
        self.cancelled = False
        self._current_mdx_model = None

    def cancel(self):
        self.cancelled = True
        if self._current_mdx_model:
            try:
                self._current_mdx_model.cleanup()
            except Exception:
                pass

    @pyqtSlot(str, str, list)
    def run(self, input_path, output_path, instruments_to_remove,
            progress_callback=None, status_callback=None, cancelled=None):

        def emit_status(msg):
            (status_callback or self.status_updated.emit)(msg)
        def emit_progress(val):
            (progress_callback or self.progress_updated.emit)(val)
        def is_cancelled():
            return cancelled() if cancelled else self.cancelled

        try:
            self.cancelled = False
            emit_status("Loading audio file...")
            waveform, sr = torchaudio.load(input_path)
            instrument = instruments_to_remove[0].lower() if instruments_to_remove else "vocals"

            # VOCALS → MDX only
            if instrument == "vocals":
                emit_status("Removing vocals with MDX...")
                emit_progress(30)
                mdx_model = MDXModelWrapper("UVR-MDX-NET-Inst_HQ_1.onnx", emit_status)
                self._current_mdx_model = mdx_model
                try:
                    sources = mdx_model(waveform, sr)
                    instrumental = sources[1] if len(sources) >= 2 else None

                    # 🔒 Silent CPU sync to prevent hum
                    if instrumental is not None:
                        _ = float(instrumental.mean())

                finally:
                    mdx_model.cleanup()

                if instrumental is None:
                    raise ValueError("Vocal removal failed")

                out_path = os.path.join(
                    get_downloads_folder(),
                    f"{os.path.splitext(os.path.basename(input_path))[0]}_no_vocals.wav"
                )
                torchaudio.save(out_path, _fix_tensor(instrumental.cpu()), sr)
                emit_progress(100)
                emit_status(f"Processing complete → {out_path}")
                self.processing_finished.emit(out_path)
                return out_path

            # OTHER stems → Demucs only
            else:
                emit_status(f"Processing {instrument} with Demucs...")
                emit_progress(30)
                model = self.model_manager.load_model_safely("htdemucs_ft", emit_status)
                model = model.to(self.model_manager.device)
                demucs_input = _fix_tensor(waveform, batch=True, device=self.model_manager.device)

                with torch.inference_mode():
                    sources = apply_model(
                        model, demucs_input,
                        device=self.model_manager.device,
                        shifts=0, overlap=0.0, split=False
                    )[0]

                out_path = os.path.join(
                    get_downloads_folder(),
                    f"{os.path.splitext(os.path.basename(input_path))[0]}_no_{instrument}.wav"
                )
                for i, name in enumerate(model.sources):
                    if name.lower() == instrument:
                        continue
                    torchaudio.save(out_path, _fix_tensor(sources[i].cpu()), sr)

                emit_progress(100)
                emit_status(f"Processing complete → {out_path}")
                self.processing_finished.emit(out_path)
                return out_path

        except Exception as e:
            print(f"[ERROR] Processor run failed: {e}\n{traceback.format_exc()}")
            emit_status(f"Error: {str(e)}")
            self.processing_finished.emit("")
            return None
