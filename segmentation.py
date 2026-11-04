from skimage import io
import numpy as np
from skimage.filters import threshold_otsu


def otsu_thresholding(image, foreground = "bright", nbins=256):
    """
    Apply Otsu's thresholding method to segment the image into foreground and background.

    Parameters:
    - image: 2D numpy array representing the grayscale image.
    - foreground: str, either "dark" or "bright", indicating which part of the image is considered foreground.
    - nbins: int, number of bins to use for histogram calculation.

    Returns:
    - mask: 2D numpy array of the same shape as input image, where foreground pixels are 1 and background pixels are 0.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")

    # Calculate Otsu's threshold
    threshold_value = threshold_otsu(image, nbins=nbins)

    # Create binary image based on the threshold and specified foreground
    if foreground == "dark":
        mask = (image <= threshold_value)
    elif foreground == "bright":
        mask = (image > threshold_value)
    else:
        raise ValueError("Foreground must be either 'dark' or 'bright'.")

    return mask, float(threshold_value)
    