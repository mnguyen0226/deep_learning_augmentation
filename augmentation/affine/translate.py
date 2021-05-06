import numpy as np

def translate(origin_img, max_tx: float = 0.3, max_ty: float = 0.3):
    """Translate an image by up to max_tx, max_ty

    Parameters
    ----------
    origin_img
        original iamge to translate
    max_tx: float, optional
        maximum horizontal distance to translate image (proportion of pixels), by default 0.3
    max_ty: float, optional
        maximum vertical distance to translate image (proportion of pixels), by default 0.3
    
    Returns
    -----
    out_img
        translated image
    """
    # find height and width
    im_height, im_width = origin_img.shape[:2]

    # set size for output image
    out_img = np.zeros_like(origin_img)

    # find translation in pixels
    tx_pix = im_height * np.random.uniform(-max_tx, max_tx)
    ty_pix = im_height * np.random.uniform(-max_ty, max_ty)

    # build translation matrix
    translate_mat = np.array(
        [
            [1, 0, tx_pix],
            [0, 1, ty_pix],
            [0, 0, 1],
        ]
    )

    # transform each pixels
    for i_coor, j_coor in np.ndindex(origin_img.shape[:2]):
        x, y, _ = translate_mat @ np.array([i_coor, j_coor, 1])

        if 0 < round(x) < origin_img.shape[0] and 0 < round(y) < origin_img.shape[1]:
            out_img[i_coor][j_coor] = origin_img[round(x)][round(y)]

    # return rotated image
    return out_img
