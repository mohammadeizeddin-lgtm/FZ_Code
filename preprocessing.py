from os import path
from pathlib import Path

import tifffile
import numpy as np


def parse_path(path_str):
    """
    Extract image information from its filename.

    Expected format:
        detector_state_chemistry_image_ID.tif

    Example:
        SE_Fresh_LFP_1.tif
    """
    name = Path(path_str).stem
    split = name.split('_')

    detector = split[0]      # SEM detector used
    state = split[1]         # Fresh or cycled
    chemistry = split[2]     # LFP, NMC, graphite, etc.
    image_ID = int(split[3]) # Image identifier

    return detector, state, chemistry, image_ID


def load_tiff(path_str):
    """Load the TIFF without changing its pixel values or dtype."""
    if not path.exists(path_str):
        raise FileNotFoundError(f"File not found: {path_str}")

    image = tifffile.imread(path_str)
    return image


def preprocess_image(
    image,
    *,
    clipping_percentile_range=None,
    gaussian_correction_filter=False,
    gaussian_background_sigma=None,
    gentle_denoising=False,
):
    """
    Return a float32 image scaled to [0, 1].

    The original input image is not modified.

    Processing order:
        1. Optional non-local means denoising.
        2. Optional Gaussian background estimation and subtraction.
        3. Optional percentile clipping.
        4. Intensity scaling to [0, 1].

    Parameters
    ----------
    image : numpy.ndarray
        A non-empty 2D grayscale image.

    clipping_percentile_range : tuple or None
        None disables percentile clipping and uses the full min/max range.
        A range such as (1, 99) enables clipping before final scaling.

    gaussian_correction_filter : bool
        Enable uneven-background correction by subtracting a broad
        Gaussian estimate. This is an approximate additive correction,
        rather than measured flat-field correction.

    gaussian_background_sigma : float or None
        Standard deviation of the background-estimation Gaussian,
        measured in pixels. Required when gaussian_correction_filter=True.

    gentle_denoising : bool
        Enable non-local means denoising before background correction.
        The noise level is estimated automatically.

    Notes
    -----
    Remove information bars before calling this function.

    Neither optional operation automatically distinguishes noise or
    imaging artefacts from genuine specimen features. Check small pores,
    cracks, boundaries and texture when selecting processing settings.

    This function is not mask-aware. Use a valid rectangular specimen
    region without unhandled annotation overlays.
    """

    # Validate the image before processing.
    image = np.asarray(image)

    if image.ndim != 2 or image.size == 0:
        raise ValueError("Expected a non-empty 2D grayscale image.")

    if np.iscomplexobj(image):
        raise ValueError("Expected real-valued image intensities.")

    print(f"Image shape: {image.shape}, Image dtype: {image.dtype}")

    # Floating point allows subtraction and fractional normalised values.
    # astype creates a separate array, preserving the original input.
    image_float = image.astype(np.float32)

    if not np.isfinite(image_float).all():
        raise ValueError(
            "Image contains NaN, infinity, or values outside float32 range."
        )

    # Validate the clipping range only when clipping is requested.
    if clipping_percentile_range is not None:
        try:
            bounds = np.asarray(
                clipping_percentile_range,
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Provide None or two percentile numbers."
            ) from exc

        if (
            bounds.shape != (2,)
            or not np.isfinite(bounds).all()
            or not 0 <= bounds[0] < bounds[1] <= 100
        ):
            raise ValueError(
                "Percentiles must satisfy 0 <= lower < upper <= 100."
            )

    # Optional step 1: non-local means denoising.
    if gentle_denoising:
        from skimage.restoration import denoise_nl_means, estimate_sigma

        # This sigma measures intensity noise, not a distance in pixels.
        # Because image_float is floating point, the estimate uses the
        # same intensity units as image_float.
        noise_sigma = float(
            estimate_sigma(image_float, channel_axis=None)
        )

        if not np.isfinite(noise_sigma) or noise_sigma < 0:
            raise ValueError(
                "Could not obtain a valid noise estimate."
            )

        # No smoothing is needed when the estimated noise is zero.
        if noise_sigma > 0:
            image_float = denoise_nl_means(
                image_float,

                # Controls smoothing strength.
                # 0.6 is a starting value requiring feature validation.
                h=0.6 * noise_sigma,

                # Account for the estimated noise in patch comparisons.
                sigma=noise_sigma,

                # Compare 5 × 5 pixel patches.
                patch_size=5,

                # Search for similar patches within this pixel distance.
                patch_distance=6,

                fast_mode=True,

                # Preserve the current intensity scale during denoising.
                preserve_range=True,

                # The input is grayscale, without a colour-channel axis.
                channel_axis=None,
            ).astype(np.float32)

            if not np.isfinite(image_float).all():
                raise ValueError(
                    "Denoising produced non-finite intensity values."
                )

    # Optional step 2: correct broad background intensity variations.
    if gaussian_correction_filter:
        if (
            gaussian_background_sigma is None
            or not np.isscalar(gaussian_background_sigma)
            or not np.isfinite(gaussian_background_sigma)
            or gaussian_background_sigma <= 0
        ):
            raise ValueError(
                "Provide a positive gaussian_background_sigma in pixels."
            )

        from scipy.ndimage import gaussian_filter

        # Broad smoothing estimates slowly varying background intensity.
        background = gaussian_filter(
            image_float,
            sigma=gaussian_background_sigma,
            mode="reflect",
        )

        # Subtract the estimate while retaining negative residuals.
        # Final normalisation handles the resulting intensity range.
        image_float = image_float - background

    # Determine scaling limits after the optional processing steps.
    if clipping_percentile_range is None:
        low, high = image_float.min(), image_float.max()
    else:
        low, high = np.percentile(image_float, bounds)

    if high <= low:
        raise ValueError(
            "Selected intensity limits have no variation; cannot normalize."
        )

    # Optional step 3: clamp values outside the percentile limits.
    # Pixels remain in the image; their extreme intensities are saturated.
    if clipping_percentile_range is not None:
        image_float = np.clip(image_float, low, high)

    # Final step: map the selected intensity range to [0, 1].
    normalised_image = (image_float - low) / (high - low)

    return normalised_image.astype(np.float32, copy=False)


def save_tiff(image, path_str):
    """
    Save the processed image as a float32 TIFF.

    Floating point preserves fractional normalised intensities.
    This array-only helper does not copy original microscope metadata.
    """
    tifffile.imwrite(
        path_str,
        image.astype(np.float32),
    )