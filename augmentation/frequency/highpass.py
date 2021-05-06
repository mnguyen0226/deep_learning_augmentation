from augmentation.frequency.lowpass import *

import math
import numpy as np
import cv2

def ideal_highpass_mask(shape, cutoff):
    """Computes Highpass lowpass mask
    
    Parameters
    ----------
    shape
        the shape of the mask to be generated
    cutoff 
        cutoff frequency
    
    Returns
    -------
    mask
        ideal highpass mask 
    """
    d0 = cutoff
    mask = 1 - ideal_lowpass_mask(shape, d0)
        
    return mask

def butterworth_highpass_mask(shape, cutoff, order = 2):    
    """Computes Butterworth highpass mask
    
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
        butterworth highpass mask
    """
    d0 = cutoff
    rows, columns = shape
    mask = np.zeros((rows, columns))
    mid_row, mid_col = int(rows / 2), int(columns / 2)
    for i in range(rows):
        for j in range(columns):
            d = math.sqrt((i - mid_row) ** 2 + (j - mid_col) ** 2)
            if d == 0:
                mask[i, j] = 0
            else:
                mask[i, j] = 1 / (1 + (d0 / d) ** (2 * order))
        
    return mask

def gaussian_highpass_mask(shape, cutoff):
    """Computes Gaussian highpass mask
    
    Parameters
    ----------
    shape
        the shape of the mask to be generated
    cutoff 
        cutoff frequency
    
    Returns
    -------
    mask
        gaussian highpass mask
    """
    d0 = cutoff
    mask = 1 - gaussian_lowpass_mask(shape, d0)

    return mask


