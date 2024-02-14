import torch
import numpy as np

def arrayStateToTensor(states: list, device: str = "cpu") -> torch.Tensor:
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        # Turn single state into a batch
        npStates = np.expand_dims(states[0], 0)
    else:
        # Transpose list of arrays into array of arrays
        npStates = np.array(states)
    return torch.from_numpy(npStates)

def dictionaryStateToTensor(states: list, device: str = "cpu") -> dict:
    """
    Convert list of encoded states into a dictionary of torch tensors suitable for the model
    and move them to the specified device (CPU or CUDA).
    :param states: list of encoded state dictionaries
    :param device: device to move tensors to (default is "cpu")
    """
    # Check input parameters
    assert isinstance(states, list)

    # Convert list of dictionaries into dictionary of arrays, skipping None values
    arrays = {key: np.array([item[key] for item in states if isinstance(item[key], np.ndarray)], copy=False) for key in states[0].keys()}

    # Convert dictionary of arrays into dictionary of tensors
    tensors = {key: torch.from_numpy(arr) for key, arr in arrays.items()}

    # Move tensors to the specified device
    tensors = {key: tensor.to(device) for key, tensor in tensors.items()}

    # Return the dictionary of tensors
    return tensors
