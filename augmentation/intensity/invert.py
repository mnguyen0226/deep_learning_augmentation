import numpy as np

def invert(origin_img):
    """Inverts image intensities.

    Parameters
    ----------
    origin_img : np.ndarray
        original image to invert

    Returns
    -------
    out_img
        inverted image
    """ 
    # if image isn't uint8, convert to it
    if not origin_img.dtype == np.uint8:
        # normalize image
        origin_img /= origin_img.max()

        # scale to 255 and convert to uint8
        orig_img = (255 * origin_img).astype(np.uint8)

    # invert image intensities and return
    out_img = 255 - origin_img

    return out_img