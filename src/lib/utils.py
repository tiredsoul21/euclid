""" Utility functions """
import operator
import numpy as np

import torch

def array_state_to_tensor(states: list) -> torch.Tensor:
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        # Turn single state into a batch
        np_states = np.expand_dims(states[0], 0)
    else:
        # Transpose list of arrays into array of arrays
        np_states = np.array(states)
    return torch.from_numpy(np_states)

def dict_state_to_tensor(states: list, device: str = "cpu") -> dict:
    """
    Convert list of encoded states into a dictionary of torch tensors suitable for the model
    and move them to the specified device (CPU or CUDA).
    :param states: list of encoded state dictionaries
    :param device: device to move tensors to (default is "cpu")
    """
    # Check input parameters
    assert isinstance(states, list)

    # Convert list of dictionaries into dictionary of arrays, skipping None values
    arrays = {key: np.array([item[key] for item in states \
                             if isinstance(item[key], np.ndarray)], copy=False) \
                                for key in states[0].keys()}

    # Convert dictionary of arrays into dictionary of tensors
    tensors = {key: torch.from_numpy(arr) for key, arr in arrays.items()}

    # Move tensors to the specified device
    tensors = {key: tensor.to(device) for key, tensor in tensors.items()}

    # Return the dictionary of tensors
    return tensors

class SegmentTree(object):
    """ Segment Tree data structure for range queries """
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        Insert and query time is O(log n).
        param capacity: size of the tree (power of 2)
        param operation: operation for combining elements (eg. sum, max)
        param neutral_element: neutral element for the operation float('-inf')
            for max, 0 for sum, ...
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0
        self._capacity = capacity
        self._operation = operation
        self._value = [neutral_element for _ in range(2 * capacity)]

    def _reduce_helper(self, start, end, node, first_node, last_node):
        # Base case: when the node is a leaf
        if start == first_node and end == last_node:
            return self._value[node]

        # Calculate the mid index (floor division)
        mid = (first_node + last_node) // 2

        # Recursive call to the operation on the range
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, first_node, mid)
        elif mid + 1 <= start:
            return self._reduce_helper(start, end, 2 * node + 1, mid + 1, last_node)
        else:
            left = self._reduce_helper(start, mid, 2 * node, first_node, mid)
            right = self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, last_node)
            return self._operation(left, right)

    def reduce(self, start=0, end=None):
        """
        Returns the operation on the given range of array elements
        param start: start index
        param end: end index
        return: operation on the given range of array elements
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val

        while idx > 1:
            idx //= 2
            self._value[idx] = self._operation( self._value[2 * idx], self._value[2 * idx + 1] )

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]

class SumSegmentTree(SegmentTree):
    """ Segment Tree data structure for range sum queries """
    def __init__(self, capacity):
        super().__init__(capacity=capacity, operation=operator.add, neutral_element=0.0 )

    def sum(self, start=0, end=None):
        """Sum of elements from index start (inclusive) to end (exclusive)"""
        return super().reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """
        Find the highest index `i` in the array such that
        `sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum`
        param perfixsum: upperbound on the sum of array prefix
        return: highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity

class MinSegmentTree(SegmentTree):
    """ Min Segment Tree data structure for range minimum queries """
    def __init__(self, capacity):
        super().__init__( capacity=capacity, operation=min, neutral_element=float('inf') )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""
        return super().reduce(start, end)
