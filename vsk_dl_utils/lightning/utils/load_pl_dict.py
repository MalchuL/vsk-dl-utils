import torch
import torch.nn as nn


def load_lighting_dict(model: nn.Module, key: str, path: str):
    """
    Loads state dict to the model by splitting key.
    Useful if you don't want to load whole model. Lightning checkpoint usually saved with additional prefixes but
    you want to load specific keys only
    Example: load_dict(unet_generator, 'generator', /home/username/checkpoints/last.pth) will initialize unet_generator
    from file /home/username/checkpoints/last.pth. Weight will be taken from generator.layer1 generator.layer2, and
     "generator." strings will be removed.
    :param model: nn.Module to load state dict.
    :param key: Keys to take and replace.
    :param path: ckpt paths to load.
    """
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if key + "." in k:
            k = str(k)
            key = str(key)
            index = k.find(key)
            new_key = k[index + len(key + "."):]
            new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
