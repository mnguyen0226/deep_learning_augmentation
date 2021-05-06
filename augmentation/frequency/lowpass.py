import math
import numpy as np
import cv2

def ideal_lowpass_mask(shape, cutoff):
    """Computes Ideal lowpass mask
    
    Parameters
    ----------
    shape
        the shape of the mask to be generated
    cutoff 
        cutoff frequency
    
    Returns
    -------
    mask
        ideal lowpass mask
    """
    d0 = cutoff
    rows, columns = shape
    mask = np.zeros((rows, columns), dtype=int)
    mid_row, mid_col = int(rows/2), int(columns/2)
    for i in range(rows):
        for j in range(columns):
            d = math.sqrt((i - mid_row)**2 + (j - mid_col)**2)
            if d <= d0:
                mask[i, j] = 1
            else:
                mask[i, j] = 0

    return mask
    
def butterworth_lowpass_mask(shape, cutoff, order = 2):
    """Computes Butterworth lowpass mask
    
    Parameters
    ----------
    shape
        the shape of the mask to be generated
    cutoff 
        cutoff frequency
    order: int: optional: 2
        order of the filter
    
    Returns
    -------
    mask
        butterworth lowpass mask
    """
    d0 = cutoff
    rows, columns = shape
    mask = np.zeros((rows, columns))
    mid_row, mid_col = int(rows / 2), int(columns / 2)
    for i in range(rows):
        for j in range(columns):
            d = math.sqrt((i - mid_row) ** 2 + (j - mid_col) ** 2)
            mask[i, j] = 1 / (1 + (d / d0) ** (2 * order))

    return mask

def gaussian_lowpass_mask(shape, cutoff):
    """Computes Gaussian lowpass mask
    
    Parameters
    ----------
    shape
        the shape of the mask to be generated
    cutoff 
        cutoff frequency
    
    Returns
    -------
    mask
        gaussian lowpass mask
    """
    d0 = cutoff
    rows, columns = shape
    mask = np.zeros((rows, columns))
    mid_row, mid_col = int(rows / 2), int(columns / 2)
    for i in range(rows):
        for j in range(columns):
            d = math.sqrt((i - mid_row) ** 2 + (j - mid_col) ** 2)
            mask[i, j] = np.exp(-(d * d) / (2 * d0 * d0))

    return mask