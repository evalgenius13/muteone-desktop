import torch

def mix_tracks(track1, track2, output_file, sample_rate=44100):
    """Mix two mono/stereo tracks and save."""
    max_len = max(track1.shape[1], track2.shape[1])
    if track1.shape[1] < max_len:
        track1 = torch.nn.functional.pad(track1, (0, max_len - track1.shape[1]))
    if track2.shape[1] < max_len:
        track2 = torch.nn.functional.pad(track2, (0, max_len - track2.shape[1]))
    mixed = track1 + track2
    import torchaudio
    torchaudio.save(output_file, mixed, sample_rate)
    return output_file
