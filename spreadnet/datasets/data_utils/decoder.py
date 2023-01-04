import io
import torch


def pt_decoder(key, data):
    """Loads PyG data (.pt format) from TAR file."""
    if not key.endswith(".pt"):
        return None
    stream = io.BytesIO(data)
    result = torch.load(stream)
    return result
