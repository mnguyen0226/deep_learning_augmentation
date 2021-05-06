import numpy as np

# https://pythontic.com/image-processing/pillow/rotate

def rotate(origin_img, max_theta = 360, mode = "non-random"):
    """Rotates an image by up to max theta degree

    Parameters:
    ----------
    origin_img
        original image to rotate
    max_theta: float, optional
        maximum positive rotation (in degree), by default 360
    mode: string, optional
        rotation randomly or with certain max_theta degree, by default "non-random"

    Returns
    -----
    out_img
        rotated image
    """
    # find height and width
    im_height, im_width = origin_img.shape[:2]

    # set size for output image
    out_img = np.zeros_like(origin_img)

    # find the rotation (in radian) - randomly
    if(mode == "non-random"):
        theta = max_theta * np.pi / 180.0
    elif (mode == "random"):
        theta = np.random.uniform(0, max_theta) * np.pi / 180.0

    # build rotation matrix
    c_x, c_y = map(lambda x: x / 2, origin_img.shape[:2])
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotate_mat = np.array(
        [
            [cos_theta, sin_theta, (1 - cos_theta) * c_x - (sin_theta * c_y)],
            [-sin_theta, cos_theta, (sin_theta * c_x) + (1 - cos_theta) * c_y],
            [0, 0, 1],
        ]
    )

    # transform each pixels
    for i_coor, j_coor in np.ndindex(origin_img.shape[:2]):
        x, y, _ = rotate_mat @ np.array([i_coor, j_coor, 1])

        if 0 < round(x) < origin_img.shape[0] and 0 < round(y) < origin_img.shape[1]:
            out_img[i_coor][j_coor] = origin_img[round(x)][round(y)]

    # return rotated image
    return out_img

