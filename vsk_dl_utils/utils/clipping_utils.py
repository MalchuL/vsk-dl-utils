import torch


def z_clip(x, z_value):
    if len(x.shape) != 4:
        raise ValueError("Only 4D tensors are supported")
    with torch.no_grad():
        std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        min = mean - std * z_value
        max = mean + std * z_value
    x = torch.clip(x, min=min, max=max)
    return x
