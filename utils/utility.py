'''
This module contains helper functions
'''
import random
import os
from typing import Iterable
import torch
import numpy as np

def set_seed(seed:int)->None:
    
    """
    Set the random seed for various libraries to ensure reproducibility.

    This function sets the seed for the random number generator used by Python's `random` module,
    NumPy, and PyTorch, including PyTorch's CUDA operations. It also configures PyTorch's
    deterministic and benchmarking settings for consistent results.

    Args:
        seed (int): The seed value to be used for random number generation.

    Example:
        >>> set_random_seed(42)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def grad_flow_dict(named_parameters:Iterable)->dict:
    """
    Computes the average gradients of the model parameters.

    This function iterates through named parameters of a model, filters out those
    that do not require gradients or are biases, and calculates the average of the
    absolute values of their gradients. It then returns a dictionary with layer names
    as keys and the corresponding average gradients as values.

    Args:
        named_parameters (Iterator[Tuple[str, torch.nn.Parameter]]): An iterator of
            tuples containing the parameter name and the parameter tensor.

    Returns:
        dict: A dictionary where keys are layer names and values are the average gradients.
    
    Example:
        >>> model = YourModel()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> loss = criterion(output, target)
        >>> loss.backward()
        >>> grad_dict = grad_flow_dict(model.named_parameters())
    """
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    
    return {layers[i]: ave_grads[i] for i in range(len(ave_grads))}

def get_numeric_part(filename):
    # Split the filename into numeric and non-numeric parts
    numeric_part = ''.join(filter(str.isdigit, filename))
    return int(numeric_part)  # Convert numeric part to integer for sorting

def sort_filenames_by_number(filenames):
    # Sort filenames based on the numeric part extracted from each filename
    sorted_filenames = sorted(filenames, key=get_numeric_part)
    return sorted_filenames