import math
import numpy as np
import cv2

def ideal_bandpass_mask(shape, cutoff):
    """Computes Ideal bandpass mask
    
    Parameters
    ----------
    shape
        the shape of the mask to be generated
    cutoff 
        cutoff frequency
    
    Returns
    -------
    mask
        ideal bandpass mask
    """
    C0 = 45
    W = 100

    d0 = C0
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=int)
    mid_row, mid_col = int(rows/2), int(cols/2)
    for i in range(rows):
        for j in range(cols):
            d = math.sqrt((i - mid_row) ** 2 + (j - mid_col)**2)
            lower_range = d0 - W/2
            upper_range = d0 + W/2
            if (d >= lower_range and d <= upper_range):
                mask[i, j] = 1
            else:
                mask[i, j] = 0
    return mask

def butterworth_bandpass_mask(shape, cutoff, order = 2):
    """Computes Butterworth bandpass mask
    
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
        butterworth bandpass mask
    """
    C0 = 70
    W = 100

    d0 = C0
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=int)
    mid_row, mid_col = int(rows/2), int(cols/2)
    for i in range(rows):
        for j in range(cols):
            d = math.sqrt((i - mid_row)**2 + (j - mid_col)**2)
            difference = (d*d - d0*d0)
            if (difference == 0):
                inner = (d*W / (0.000001))
                deno = 1 + (inner) ** (2*order)
                mask[i, j] = 1 - (1/deno)
            else:
                inner = (d*W / (difference))
                deno = 1 + (inner) ** (2*order)
                mask[i, j] = 1 - (1/deno)

    return mask

def gaussian_bandpass_mask(shape, cutoff):
    """Computes Gaussian bandpass mask
    
    Parameters
    ----------
    shape
        the shape of the mask to be generated
    cutoff 
        cutoff frequency
    
    Returns
    -------
    mask
        gaussian bandpass mask
    """
    C0 = 70
    W = 200

    d0 = C0
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=int)
    mid_row, mid_col = int(rows/2), int(cols/2)
    for i in range(rows):
        for j in range(cols):
            d = math.sqrt((i - mid_row)**2 + (j - mid_col)**2)
            if (d == 0):
                mask[i, j] = np.exp(-((d*d - d0*d0) / 0.000001*W)**2)
            else:
                mask[i, j] = np.exp(-((d*d - d0*d0) / d*W)**2)
    return mask