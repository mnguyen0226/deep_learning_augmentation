import numpy as np


def hist_equalization(origin_img):
    """Performs histogram equalization on the provided image

    Parameters
    ----------
    origin_img : np.ndarray
        image to histogram equalize

    Returns
    -------
    out_img
        histogram equalized image
    """

    # create normalized cumulative histogram
    hist_arr = np.bincount(origin_img.ravel())
    cum_hist_arr = np.cumsum(hist_arr / np.sum(hist_arr))

    # generate transformation lookup table
    transform_lut = np.floor(255 * cum_hist_arr).astype(np.uint8)

    # perform lookups and return resulting histogram equalized image
    out_img = transform_lut[origin_img]

    return out_img