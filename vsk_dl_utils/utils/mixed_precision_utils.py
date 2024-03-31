from typing import Union

import torch
from torch import nn


def get_torch_dtype(precision: Union[int, str]):
    precision = str(precision)
    if precision == '16':
        return torch.float16
    elif precision == '32':
        return torch.float32
    elif precision in ['b16', 'bfloat16']:
        return torch.bfloat16
    else:
        raise ValueError(f'Precision {precision} is not supported')


def convert_tensor_dtype(tensor: torch.Tensor, precision):
    dtype = get_torch_dtype(precision)
    return tensor.to(dtype=dtype)


def convert_model_dtype(model: nn.Module, precision: Union[int, str]):
    dtype = get_torch_dtype(precision)
    if dtype == torch.float16:
        return model.half()
    elif dtype == torch.float32:
        return model.float()
    elif dtype == torch.bfloat16:
        return model.to(dtype=dtype)
    else:
        raise ValueError(f'Precision {precision} is not supported')
