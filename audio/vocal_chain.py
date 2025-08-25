import numpy as np

class VocalChain:
    """Placeholder vocal FX chain with EQ, Compression, and Reverb controls."""

    def __init__(self):
        self.enabled = True

        # EQ
        self.eq_gain_db = 0.0

        # Compressor
        self.comp_threshold = -12.0
        self.comp_ratio = 2.0

        # Reverb
        self.reverb_amount = 0.2

    def set_params(self, eq_gain_db=None, comp_threshold=None,
                   comp_ratio=None, reverb_amount=None, enabled=None):
        """Update parameters from UI sliders."""
        if eq_gain_db is not None:
            self.eq_gain_db = eq_gain_db
        if comp_threshold is not None:
            self.comp_threshold = comp_threshold
        if comp_ratio is not None:
            self.comp_ratio = comp_ratio
        if reverb_amount is not None:
            self.reverb_amount = reverb_amount
        if enabled is not None:
            self.enabled = enabled

    def process(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """
        Placeholder processing chain.
        Currently passes audio through unchanged.
        Later you’ll add DSP here.
        """
        if not self.enabled:
            return audio_data

        # (Future: apply EQ → Compression → Reverb here)
        return audio_data
