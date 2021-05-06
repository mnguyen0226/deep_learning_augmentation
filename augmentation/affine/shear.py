import numpy as np

def vertical_shear(origin_img, sv = 0.5):
    """Shears an image vertically

    Parameters:
    ----------
    origin_img
        original image to shear iamge vertically
    sv: float, optional
        shearing percentage

    Returns:
    -----
    out_img
        vertically sheared image
    """
    # find height and width
    im_height, im_width = origin_img.shape[:2]

    # set size for output image
    out_img = np.zeros_like(origin_img)

    # build a vertical shear matrix 
    vert_shear_matrix =  np.array([[1, sv, 0], [0, 1, 0], [0, 0, 1]])

    # transform each pixels
    for i_coor, j_coor in np.ndindex(origin_img.shape[:2]):
        x, y, _ = vert_shear_matrix @ np.array([i_coor, j_coor, 1])

        if 0 < round(x) < origin_img.shape[0] and 0 < round(y) < origin_img.shape[1]:
            out_img[i_coor][j_coor] = origin_img[round(x)][round(y)]

    # return rotated image
    return out_img

def horizontal_shear(origin_img, sh = 0.5):
    """Shears an image horizontal

    Parameters:
    ----------
    origin_img
        original image to shear iamge horizontally
    sv: float, optional
        shearing percentage

    Returns:
    -----
    out_img
        horizontally sheared image
    """
    # find height and width
    im_height, im_width = origin_img.shape[:2]

    # set size for output image
    out_img = np.zeros_like(origin_img)

    # build a vertical shear matrix 
    hori_shear_matrix =  np.array([[1, 0, 0], [sh, 1, 0], [0, 0, 1]])

    # transform each pixels
    for i_coor, j_coor in np.ndindex(origin_img.shape[:2]):
        x, y, _ = hori_shear_matrix @ np.array([i_coor, j_coor, 1])

        if 0 < round(x) < origin_img.shape[0] and 0 < round(y) < origin_img.shape[1]:
            out_img[i_coor][j_coor] = origin_img[round(x)][round(y)]

    # return rotated image
    return out_img