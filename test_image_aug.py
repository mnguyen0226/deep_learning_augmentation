"""
Tryout script allowing user to test with all image augmentation method in one image
python ./test_image_aug.py
"""
from augmentation.affine.flip import *
from augmentation.affine.rotate import rotate
from augmentation.affine.translate import translate
from augmentation.affine.shear import *
from augmentation.frequency.frequency_filter import freq_filter
from augmentation.intensity.invert import invert
from augmentation.intensity.hist_equalization import hist_equalization
from augmentation.intensity.amf import amf
from augmentation.edge_detection.canny_edge import canny
from augmentation.frequency.frequency_filter import *

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def try_flip_vert():
    # Read in image of a girl
    origin_img = cv.imread("girl.tif")
    cv.imshow("Original", origin_img)
    print(f"Shape of the original image is {origin_img.shape}")

    # Flip vertically
    fv_img = flip_vertical(origin_img)
    cv.imshow("Vertical Flipped Image", fv_img)
    print(f"Shape of the augmented image is {fv_img.shape}")

    cv.waitKey(0)
    cv.destroyAllWindows() # Allow to press enter to delete all

def try_flip_hori():
    # Read in image of a girl
    origin_img = cv.imread("girl.tif")
    cv.imshow("Original", origin_img)
    print(f"Shape of the original image is {origin_img.shape}")

    # Flip horizontally 
    fh_img = flip_horizontal(origin_img)
    cv.imshow("Horizontal Flipped Image", fh_img)
    print(f"Shape of the augmented image is {fh_img.shape}")

    cv.waitKey(0)
    cv.destroyAllWindows() # Allow to press enter to delete all

def try_rand_rotate():
    # Read in image of a girl
    origin_img = cv.imread("girl.tif")
    cv.imshow("Original", origin_img)
    print(f"Shape of the original image is {origin_img.shape}")

    # Rotate image 
    rt_img = rotate(origin_img, 45, "random")
    cv.imshow("Rotational Image", rt_img)
    print(f"Shape of the augmented image is {rt_img.shape}")

    cv.waitKey(0)
    cv.destroyAllWindows() # Allow to press enter to delete all

def try_translate():
    # Read in image of a girl
    origin_img = cv.imread("girl.tif")
    cv.imshow("Original", origin_img)
    print(f"Shape of the original image is {origin_img.shape}")

    # Translation image
    trans_img = translate(origin_img)
    cv.imshow("Translation Image", trans_img)
    print(f"Shape of the augmented image is {trans_img.shape}")

    cv.waitKey(0)
    cv.destroyAllWindows() # Allow to press enter to delete all

def try_shear_vert():
    # Read in image of a girl
    origin_img = cv.imread("girl.tif")
    cv.imshow("Original", origin_img)
    print(f"Shape of the original image is {origin_img.shape}")

    # Shear image vertically
    vshear_img = vertical_shear(origin_img)
    cv.imshow("Vertical Sheared Image", vshear_img)
    print(f"Shape of the augmented image is {vshear_img.shape}")

    cv.waitKey(0)
    cv.destroyAllWindows() # Allow to press enter to delete all

def try_shear_hori():
    # Read in image of a girl
    origin_img = cv.imread("girl.tif")
    cv.imshow("Original", origin_img)
    print(f"Shape of the original image is {origin_img.shape}")

    # Shear image horizontally
    hshear_img = horizontal_shear(origin_img)
    cv.imshow("Horizontal Shear Image", hshear_img)
    print(f"Shape of the augmented image is {hshear_img.shape}")

    cv.waitKey(0)
    cv.destroyAllWindows() # Allow to press enter to delete all

def try_gauss_lpf():
    # Read in image of a girl
    origin_img = cv.imread("girl.tif")
    cv.imshow("Original", origin_img)
    print(f"Shape of the original image is {origin_img.shape}")

    # Gaussian lowpass filter
    gauss_lpf = freq_filter(origin_img, 8, 2, "gaussian_lpf")
    cv.imshow("Gaussian LPF", gauss_lpf)
    print(f"Shape of the augmented image is {gauss_lpf.shape}")

    cv.waitKey(0)
    cv.destroyAllWindows() # Allow to press enter to delete all

def try_gauss_hpf():
    # Read in image of a girl
    origin_img = cv.imread("girl.tif")
    cv.imshow("Original", origin_img)
    print(f"Shape of the original image is {origin_img.shape}")

    # Gaussian lowpass filter
    gauss_hpf = freq_filter(origin_img, 8, 2, "gaussian_hpf")
    cv.imshow("Gaussian HPF", gauss_hpf)
    print(f"Shape of the augmented image is {gauss_hpf.shape}")

    cv.waitKey(0)
    cv.destroyAllWindows() # Allow to press enter to delete all

def try_ideal_lpf():
    # Read in image of a girl
    origin_img = cv.imread("girl.tif")
    cv.imshow("Original", origin_img)
    print(f"Shape of the original image is {origin_img.shape}")

    # Ideal lowpass filter
    ideal_lpf = freq_filter(origin_img, 8, 2, "ideal_lpf")
    cv.imshow("Ideal LPF", ideal_lpf)
    print(f"Shape of the augmented image is {ideal_lpf.shape}")

    cv.waitKey(0)
    cv.destroyAllWindows() # Allow to press enter to delete all

def try_ideal_hpf():
    # Read in image of a girl
    origin_img = cv.imread("girl.tif")
    cv.imshow("Original", origin_img)
    print(f"Shape of the original image is {origin_img.shape}")

    # Ideal lowpass filter
    ideal_hpf = freq_filter(origin_img, 8, 2, "ideal_hpf")
    cv.imshow("Ideal HPF", ideal_hpf)
    print(f"Shape of the augmented image is {ideal_hpf.shape}")

    cv.waitKey(0)
    cv.destroyAllWindows() # Allow to press enter to delete all

def try_butterworth_hpf():
    # Read in image of a girl
    origin_img = cv.imread("girl.tif")
    cv.imshow("Original", origin_img)
    print(f"Shape of the original image is {origin_img.shape}")

    # Butterworth highpass filter
    butterworth_hpf = freq_filter(origin_img, 8, 2, "butterworth_hpf")
    cv.imshow("Butterworth LPF", butterworth_hpf)
    print(f"Shape of the augmented image is {butterworth_hpf.shape}")

    cv.waitKey(0)
    cv.destroyAllWindows() # Allow to press enter to delete all

def try_butterworth_lpf():
    # Read in image of a girl
    origin_img = cv.imread("girl.tif")
    cv.imshow("Original", origin_img)
    print(f"Shape of the original image is {origin_img.shape}")

    # Butterworth lowpass filter
    butterworth_lpf = freq_filter(origin_img, 8, 2, "butterworth_lpf")
    cv.imshow("Butterworth LPF", butterworth_lpf)
    print(f"Shape of the augmented image is {butterworth_lpf.shape}")

    cv.waitKey(0)
    cv.destroyAllWindows() # Allow to press enter to delete all

def try_invert():
    # Read in image of a girl
    origin_img = cv.imread("girl.tif")
    cv.imshow("Original", origin_img)
    print(f"Shape of the original image is {origin_img.shape}")

    # Invert intensity
    invert_im = invert(origin_img)
    cv.imshow("Inverted", invert_im)
    print(f"Shape of the augmented image is {invert_im.shape}")

    cv.waitKey(0)
    cv.destroyAllWindows() # Allow to press enter to delete all

def try_hist_eq():
    # Read in image of a girl
    origin_img = cv.imread("girl.tif")
    cv.imshow("Original", origin_img)
    print(f"Shape of the original image is {origin_img.shape}")

    # Hist-equalization
    hist_eq = hist_equalization(origin_img)
    cv.imshow("Hist Equalization", hist_eq)
    print(f"Shape of the augmented image is {hist_eq.shape}")

    cv.waitKey(0)
    cv.destroyAllWindows() # Allow to press enter to delete all

def try_amf():
    # Read in image of a girl
    origin_img = cv.imread("girl.tif")
    cv.imshow("Original", origin_img)
    print(f"Shape of the original image is {origin_img.shape}")

    # AMF
    amf_im = amf(origin_img)
    cv.imshow("AMF", amf_im)
    print(f"Shape of the augmented image is {amf.shape}")

    cv.waitKey(0)
    cv.destroyAllWindows() # Allow to press enter to delete all

def try_canny_ed():
    # Read in image of a girl
    origin_img = cv.imread("girl.tif")
    cv.imshow("Original", origin_img)
    print(f"Shape of the original image is {origin_img.shape}")

    # Canny Edge Detection
    canny_im = canny(origin_img)
    cv.imshow("Canny Edge Detection", np.float64(canny_im))
    print(f"Shape of the augmented image is {canny_im.shape}")

    cv.waitKey(0)
    cv.destroyAllWindows() # Allow to press enter to delete all

if __name__ == "__main__":
    try_flip_vert()
    # try_flip_hori()
    # try_rand_rotate()
    # try_translate()
    # try_shear_vert()
    # try_shear_hori()
    # try_gauss_lpf()
    # try_gauss_hpf()
    # try_ideal_lpf()
    # try_ideal_hpf()
    # try_butterworth_hpf()
    # try_butterworth_lpf()
    # try_invert()
    # try_hist_eq()
    # try_amf()
    # try_canny_ed()

