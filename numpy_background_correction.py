import numpy as np

from imagingtester import MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE


def numpy_background_correction(
    dark, data, flat, clip_min=MINIMUM_PIXEL_VALUE, clip_max=MAXIMUM_PIXEL_VALUE
):
    norm_divide = np.subtract(flat, dark)
    norm_divide[norm_divide == 0] = MINIMUM_PIXEL_VALUE
    np.subtract(data, dark, out=data)
    np.true_divide(data, norm_divide, out=data)
    np.clip(data, clip_min, clip_max, out=data)
