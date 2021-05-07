from Minh.DIP.augmentation.frequency.lowpass import *
from Minh.DIP.augmentation.frequency.bandpass import *
from Minh.DIP.augmentation.frequency.highpass import *

import math
import numpy as np
import cv2


def freq_filter(origin_img, cutoff = 8, order = 2, filter_name = "gaussian_lpf"):
    """Performs frequency filtering on an input image
       
    Parameters
    ----------
    origin_im 
        original input image
    cutoff
        cutoff frequency
    order
        order of butterworth filtering
    filter_name
        type of filter

    Returns
    -------
    filtered_image
        filtered output image
    """
    origin_img = rgb2gray(origin_img)

    shape = np.shape(origin_img)
    
    if(len(shape) == 1):
      return
    
#    print(f"TESTING: Shape is {shape}")

    # get the mask (write your code in functions provided above) the functions can be called by self.filter(shape, cutoff, order)
    if (filter_name == "ideal_lpf"):
        filter_mask = ideal_lowpass_mask(shape, cutoff)

    elif (filter_name == "butterworth_lpf"):
        filter_mask = butterworth_lowpass_mask(shape, cutoff, order)

    elif (filter_name == "gaussian_lpf"):
        filter_mask = gaussian_lowpass_mask(shape, cutoff)
        
    elif (filter_name == "ideal_bpf"):
        filter_mask = ideal_bandpass_mask(shape, cutoff)

    elif (filter_name == "butterworth_bpf"):
        filter_mask = butterworth_bandpass_mask(shape, cutoff, order)

    elif (filter_name == "gaussian_bpf"):
        filter_mask = gaussian_bandpass_mask(shape, cutoff)

    elif (filter_name == "ideal_hpf"):
        filter_mask = ideal_highpass_mask(shape, cutoff)
    
    elif (filter_name == "butterworth_hpf"):
        filter_mask = butterworth_highpass_mask(shape, cutoff, order)

    elif (filter_name == "gaussian_hpf"):
        filter_mask = gaussian_highpass_mask(shape, cutoff)
    
    # compute the fft of the image
    fft = np.fft.fft2(origin_img)

    # shift the fft to center the low frequencies
    shift_fft = np.fft.fftshift(fft)
    mag_dft = np.log(np.abs(shift_fft))
    dft = post_process_image(mag_dft)

    # filter the image frequency based on the mask (Convolution theorem)
    filtered_image = np.multiply(filter_mask, shift_fft)
    mag_filtered_dft = np.log(np.abs(filtered_image)+1)
    filtered_dft = post_process_image(mag_filtered_dft)

    # compute the inverse shift
    shift_ifft = np.fft.ifftshift(filtered_image)

    # compute the inverse fourier transform
    ifft = np.fft.ifft2(shift_ifft)

    # compute the magnitude
    mag = np.abs(ifft)

    # full contrast stretch
    filtered_image = post_process_image(mag)

    # convert two 3 dim
    if (filtered_image.ndim == 2):
      filtered_image = np.expand_dims(filtered_image, axis = 2)
    
    return filtered_image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def post_process_image(image):
    """Post process the image to create a full constrast stretch of the image
    
    Parameters
    ----------
    image
        image after inverse fourier transform

    Returns
    -------
    np.uint8
        The image with full contrast stretch
    """
    a = 0
    b = 255
    c = np.min(image)
    d = np.max(image)
    rows, cols = np.shape(image)
    out_im = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            if(d-c == 0): # avoid to have d-c = 0 which give infinite answer
                out_im[i, j] = ((b-a)/ 0.000001) * (image[i, j] - c) + a
            else:
                out_im[i, j] = ((b-a)/ (d-c)) * (image[i, j] - c) + a

    return np.uint8(out_im)    