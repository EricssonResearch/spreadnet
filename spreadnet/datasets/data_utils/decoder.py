import io
import torch


def pt_decoder(key, data):
    if not key.endswith(".pt"):
        return None
    stream = io.BytesIO(data)
    result = torch.load(stream)
    return result
