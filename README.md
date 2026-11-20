# SEM image analysis workflow

Process raw SEM TIFFs through metadata extraction, preprocessing, segmentation and particle-size analysis while preserving the original files.

## Capabilities

| Stage | What it does | Main outputs |
| --- | --- | --- |
| **1. Prepare raw images** | Extract acquisition metadata and pixel calibration; remove supported information bars; flag uncertain crops for review. Includes a TESCAN reader and generic TIFF/text metadata reading for other manufacturers. | Cropped TIFFs, review previews, audit JSON and `metadata.csv` |
| **2. Preprocess** | Scale intensities to 0–1 as float32. Optionally apply non-local means denoising, broad Gaussian background correction and percentile clipping. | Preprocessed TIFFs, audit JSON and `preprocessing_metadata.csv` |
| **3. Segment** | Choose Otsu or Chan–Vese; select bright or dark foreground. Retain Chan–Vese's floating-point level set for contour measurements. | Binary TIFF masks, colour overlays and `segmentation_history.csv` |
| **4. Measure particle sizes** | Measure Chan–Vese zero contours during segmentation, or separately count pixels in binary masks from either method. Export per-object areas, equivalent diameters, labelled images, distribution plots and statistics. | Separate contour and binary particle tables and PSD summaries |

All stages support percentage progress, incremental updates and reuse of verified outputs. Earlier configurations are retained.

## Setup

Use **Python 3.12+** and keep all supplied `.py` files together, including `sem_progress.py`.

```sh
python -m pip install -r requirements.txt
```

## Run the complete workflow

```python
from pathlib import Path
from sem_extract_info import process_sem_images
from preprocessing import preprocess_images
from segmentation import segment_images
from extracting_discriptors import particle_size_distribution

base = Path(r"C:\SEM")
processed = base / "processed"
preprocessed = processed / "preprocessed"

# 1. Extract metadata and prepare approved specimen images.
print(process_sem_images(base / "raw", processed, recursive=True))

# 2. Normalise only; optional preprocessing defaults off.
print(preprocess_images(processed, preprocessed))

# 3. Otsu segmentation, then Chan–Vese with contour-based sizing.
print(segment_images(preprocessed, method="OTSU"))
print(segment_images(
    preprocessed, method="CHAN_VESE", foreground="bright",
    chan_vese_parameters={"mu": 0.1},
    percentile_range=(10, 90),
))

# 4. Separate binary-mask sizing for both segmentation methods.
print(particle_size_distribution(
    preprocessed / "segmented", percentile_range=(10, 90),
))
```

## Main options

- **Inputs:** one TIFF, multiple paths or a folder for extraction/preprocessing. `recursive=True` includes subfolders; preprocessing automatically selects approved extraction outputs.
- **Crop review:** approve boundaries with `crop_rows={"example.tif": 768}`. Exclusion masks are supported during extraction; masked cases require mask-aware downstream processing and are held for review.
- **Preprocessing:** `clipping_percentile_range=(1, 99)` enables clipping. `gentle_denoising` and `gaussian_correction_filter` are independent options; `background_sigma` is in pixels.
- **Segmentation:** choose `foreground="bright"` or `"dark"`. `measure_particles=False` disables Chan–Vese contour sizing. Otsu only segments here.
- **Distributions:** `area_basis="solid"` excludes holes; `"envelope"` includes them. Defaults: `exclude_border=True`, `histogram_bins=30`. Use `methods="OTSU"` or `"CHAN_VESE"` to restrict binary measurements.
- **Progress:** enabled by default. Disable with `show_progress=False` or CLI `--no-progress`. Percentages count completed images/records.

## Particle-size results

Sizes are **number-weighted, 2D equivalent circular diameters**: `sqrt(4 × area / π)`. Outputs include a histogram, cumulative plot, count, mean, median, sample standard deviation, min/max, P10/P50/P90 and configurable percentiles. `percentile_range` summarises sizes; it does **not** clip them.

Under `preprocessed/segmented/`:

| File | Contents |
| --- | --- |
| `contour_particles.csv` / `contour_psd_summary.csv` | Chan–Vese contour measurements and distribution summaries |
| `binary_descriptors/` | `binary_particles.csv`, `binary_psd_summary.csv`, labelled images and plots |
| `segmentation_history.csv` | Per-segmentation records with separate `contour_psd_*` and `binary_psd_*` statistics |

Measurement records include plots, labelled images and retained contour coordinates where applicable. Summary tables retain earlier settings; segmentation history shows the latest summary per route.

Calibration comes **per image** from CSV fields `pixel_size_x_nm` and `pixel_size_y_nm`. Sizes use µm when calibrated, otherwise explicitly labelled pixel units. Touching particles remain one object. Border-cut objects are excluded by default; open contours have no area measurement.

**Before analysis:** review warnings and validate boundaries and fine features. Raw cropping supports single-page 8/16-bit grayscale TIFFs; unsupported crops go to review. Non-TIFF inputs and universal private-vendor decoding are unsupported. Keep originals, CSVs and audit JSON, and close CSVs in Excel before running.
