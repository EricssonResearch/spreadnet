import json
import numpy as np
import torch
import io


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def pt_encoder(data):
    buffer = io.BytesIO()
    torch.save(data["pt"], buffer)
    data["pt"] = buffer.getbuffer()
    return data
