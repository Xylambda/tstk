""" 
Low-level NumPy operations that are usefull for the whole library.
"""
import numpy as np
from typing import Tuple, Union


def np_rolling(
    array: np.ndarray,
    window_shape: Union[int, Tuple[int]],
    axis: Union[int, Tuple[int]]=0
):
    """Compute a sliding window view of the given array.

    The function assumes that rows represent observations and columns represent
    different time series.

    Parameters
    ----------
    array : numpy.array
        Array to apply rolling window to.
    window_shape : int
        Size of window over each axis that takes part in the sliding window. If 
        axis is not present, must have same length as the number of input array 
        dimensions. Single integers i are treated as if they were the tuple 
        (i,).
    axis : int or tuple of int, optional
        Axis or axes along which the sliding window is applied. By default, 
        the sliding window is applied to all axes and window_shape[i] will 
        refer to axis i of x. If axis is given as a tuple of int, 
        window_shape[i] will refer to the axis axis[i] of x. Single integers 
        i are treated as if they were the tuple (i,).

    Returns
    -------
    sliding_view : numpy.array
        Sliding window view of the given array.
    """
    arr = array.__array__() # DISC: not really needed

    # deal with 1d column-array
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.flatten()

    sliding_view = np.lib.stride_tricks.sliding_window_view(
        x=arr,
        window_shape=window_shape,
        axis=axis
    )

    return array.__array_wrap__(sliding_view)