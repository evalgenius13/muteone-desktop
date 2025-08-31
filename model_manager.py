import torch
from demucs.pretrained import get_model


class ModelManager:
    def __init__(self, device=None):
        """
        Manages which device Demucs runs on.
        - Prefers user-specified device if given
        - Else prefers MPS (Apple GPU on Mac)
        - Else falls back to CPU
        Note: MDX models are handled separately in processor.py and always run on CPU.
        """
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"[DEBUG] ModelManager using device: {self.device} for Demucs")

    def load_model_safely(self, model_name, status_callback=None):
        """
        Loads a Demucs model by name, with error handling and optional UI status updates.
        """
        try:
            if status_callback:
                status_callback(f"Loading model {model_name} on {self.device}...")
            # MDX handled separately (processor.MDXModelWrapper)
            if model_name.startswith("mdx"):
                return model_name
            model = get_model(model_name)
            return model
        except Exception as e:
            if status_callback:
                status_callback(f"Error loading model {model_name}: {e}")
            raise
