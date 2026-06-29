import numpy as np
import torch
from paths import data_path

def get_batch(split, seq_len, batch_size, device="cpu"):
    data = np.memmap(data_path(split), dtype=np.uint16, mode="r")
    HIGH = len(data) - seq_len
    ix = torch.randint(HIGH, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i+seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1 : i+1+seq_len].astype(np.int64)) for i in ix])
    if "cuda" in str(device):
        # pin_memory + non_blocking lets the host->GPU copy overlap with compute
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y