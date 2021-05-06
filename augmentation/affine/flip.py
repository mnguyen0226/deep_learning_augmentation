import numpy as np

def flip_vertical(origin_img):
    """Flips an image vertically

    Parameters:
    ----------
    origin_img
        original image to flip vertically

    Returns:
    -----
    out_img
        vertically flipped image
    """
    # find height and width
    im_height, im_width = origin_img.shape[:2]

    # build a vertical flip matrix
    vert_flip_mat = np.array([[-1, 0, im_height - 1], [0, 1, 0], [0, 0, 1]])

    # set size for output image
    out_img = np.zeros_like(origin_img)

    # transform each pixels
    for i_coor, j_coor in np.ndindex(origin_img.shape[:2]):
        x, y, _ = vert_flip_mat @ np.array([i_coor, j_coor, 1])

        if 0 < round(x) < origin_img.shape[0] and 0 < round(y) < origin_img.shape[1]:
            out_img[i_coor][j_coor] = origin_img[round(x)][round(y)]

    # Return flipped image
    return out_img

def flip_horizontal(origin_img):
    """FLips an image horizontally

    Parameters:
    ----------
    origin_img
        original image to flip horizontally

    Returns:
    -----
    out_img
        horizontally flipped image
    """
    # find height and width
    im_height, im_width = origin_img.shape[:2]

    # build a vertical flip matrix
    hori_flip_mat = np.array([[1, 0, 0], [0, -1, im_width - 1], [0, 0, 1]])

    # set size for output image
    out_img = np.zeros_like(origin_img)

    # transform each pixels
    for i_coor, j_coor in np.ndindex(origin_img.shape[:2]):
        x, y, _ = hori_flip_mat @ np.array([i_coor, j_coor, 1])

        if 0 < round(x) < origin_img.shape[0] and 0 < round(y) < origin_img.shape[1]:
            out_img[i_coor][j_coor] = origin_img[round(x)][round(y)]

    # return flipped image
    return out_img

# Ideas: Rotation
"""
Rotation: http://datahacker.rs/003-how-to-resize-translate-flip-and-rotate-an-image-with-opencv/
Rotation: https://towardsdatascience.com/image-geometric-transformation-in-numpy-and-opencv-936f5cd1d315
Rotation: https://stackabuse.com/affine-image-transformations-in-python-with-numpy-pillow-and-opencv/

=> Use SHAP
"""