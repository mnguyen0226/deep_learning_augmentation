import math
import cv2 as cv
import numpy as np
from scipy.ndimage.filters import convolve
from scipy import ndimage
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


def get_sobel_kernel(im):
    """Takes in smoothed images then do matrix multiplication with Sobel kernel to get the gradient magnitude and angle
    
    Parameters
    ----------
    im
        input image

    Returns
    -------    
    gradient
        gradient magnitude of sobel filter
    theta
        the angle of sobel filter
    """
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    image_x = ndimage.filters.convolve(im, kernel_x)
    image_y = ndimage.filters.convolve(im, kernel_y)
    
    gradient = np.hypot(image_x, image_y) 
    gradient = gradient / gradient.max() * 255
    theta = np.arctan2(image_y, image_x)
    
    return (gradient, theta)

def non_max_supression(grad_im, theta):
    """Goes thru all points on the gradient intensity matrix to finds the pixels with the max value in the edge directions
    
    Parameters
    ----------
    grad_im
        gradient magnitude from sobel filter
    theta
        angle from sobel filter

    Returns
    -------
    out_im
        output image
    """
    row, col = grad_im.shape # M, N
    out_im = np.zeros((row, col), dtype=np.int32)
    angle = theta*180. / np.pi
    angle[angle < 0] += 180

    for i in range(1,row-1):
        for j in range(1,col-1):
            q = 255
            r = 255
                
            # for angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = grad_im[i, j+1]
                r = grad_im[i, j-1]
            # for angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = grad_im[i+1, j-1]
                r = grad_im[i-1, j+1]
            # for angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = grad_im[i+1, j]
                r = grad_im[i-1, j]
            # for angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = grad_im[i-1, j-1]
                r = grad_im[i+1, j+1]

            if (grad_im[i,j] >= q) and (grad_im[i,j] >= r):
                out_im[i,j] = grad_im[i,j]
            else:
                out_im[i,j] = 0
    return out_im

def double_threshold(im, ltr, htr):
    """Double threshold aim to identify strong, weak, and non-relevant pixels

    Parameters
    ----------
    im
        input image
    ltr
        low threshold ratio
    htr 
        high threshold ratio

    Returns
    -------
    weak
        weak pixel that has intensity value that is not enough to be considered strong ones, but not non-relevant
    strong
        pixel with very high intensity
    res
        non-relevant pixel, not high, not low
    """
    high_thres = im.max() * htr
    low_thres = high_thres * ltr
    
    row, col = im.shape
    res = np.zeros((row,col), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(im >= high_thres)
    zeros_i, zeros_j = np.where(im < low_thres)
    
    weak_i, weak_j = np.where((im <= high_thres) & (im >= low_thres))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def hysteresis(im, weak, strong=255):
    """Transforms weak pixel into strong ones if at least 1 pixel around the one being processed is a strong one 

    Parameters
    ----------
    im 
        input image
    weak
        weak pixel that has intensity value that is not enough to be considered strong ones, but not non-relevant
    strong
        pixel with very high intensity
    
    Returns
    -------
    im
        output result image
    """
    row, col = im.shape  
    
    for i in range(1, row-1):
        for j in range(1, col-1):
            if (im[i,j] == weak):
                if ((im[i+1, j-1] == strong) or (im[i+1, j] == strong) or (im[i+1, j+1] == strong)
                    or (im[i, j-1] == strong) or (im[i, j+1] == strong)
                    or (im[i-1, j-1] == strong) or (im[i-1, j] == strong) or (im[i-1, j+1] == strong)):
                    im[i, j] = strong
                else:     
                   im[i, j] = 0
    return im

def rgb2gray(rgb):
    """Transform 3 colors channels image to grayscale
    
    Parameters
    ----------
    rgb
        input image rgb
    
    Returns
    -------
    np.dot
        grayscale image
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def canny(origin_img):
    """Applies Canny Edge Detection Steps

    Parameters
    ----------
    origin_img
        input original image
    
    Returns
    -------
    im_final
        final output edge-detected image
    """
    origin_img = rgb2gray(origin_img)

    origin_img = origin_img/255.0

    # blur image with Gaussian filter with sigma = 2,8,16
    filtered_im1 = cv.GaussianBlur(origin_img, (0,0), 2)
    filtered_im2 = cv.GaussianBlur(origin_img, (0,0), 8)
    filtered_im3 = cv.GaussianBlur(origin_img, (0,0), 16)


    # apply Canny edge detectors with the scales you choose to detect edges
    grad_mag1, angle1 = get_sobel_kernel(filtered_im1)
    grad_mag2, angle2 = get_sobel_kernel(filtered_im2)
    grad_mag3, angle3 = get_sobel_kernel(filtered_im3)

    # apply non-max-suppression
    im_nms1 = non_max_supression(grad_mag1, angle1)
    im_nms2 = non_max_supression(grad_mag2, angle2)
    im_nms3 = non_max_supression(grad_mag3, angle3)

    im_thres1, weak1, strong1 = double_threshold(im_nms1, ltr=0.07, htr=0.19)
    im_thres2, weak2, strong2 = double_threshold(im_nms2, ltr=0.07, htr=0.19)
    im_thres3, weak3, strong3 = double_threshold(im_nms3, ltr=0.07, htr=0.19)
    
    im_hys1 = hysteresis(im_thres1, weak1, strong1)
    im_hys2 = hysteresis(im_thres2, weak2, strong2)
    im_hys3 = hysteresis(im_thres3, weak3, strong3)

    im_final = im_hys1 + im_hys2 + im_hys3
    
    im_final = im_final/3.0

    # convert two 3 dim
    if (im_final.ndim == 2):
      im_final = np.expand_dims(im_final, axis = 2)

    return im_final




