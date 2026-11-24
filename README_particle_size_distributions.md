# SEM particle-size distributions

Keep `segmentation.py`, `extracting_discriptors.py`, `sem_progress.py` and `sem_paths.py` together. The second
filename preserves the spelling requested. Dependencies: NumPy, SciPy,
scikit-image, Pillow and Matplotlib.

Progress is enabled by default; use `show_progress=False` in either batch call,
or `--no-progress` on the command line, to disable it. See `README_progress.md`.

## Run the two routes

```python
from pathlib import Path
from segmentation import segment_images
from extracting_discriptors import particle_size_distribution

input_folder = Path(r"C:\SEM\processed\preprocessed")

# Otsu: segmentation only; no size measurements here.
print(segment_images(images=input_folder, method="OTSU"))

# Chan–Vese: retain its continuous level set, measure its zero contours,
# and export contour-based particle-size statistics and plots.
print(segment_images(
    images=input_folder,
    method="CHAN_VESE",
    chan_vese_parameters={"mu": 0.1, "max_num_iter": 500, "tol": 0.001},
    measure_particles=True,
    percentile_range=(10, 90),
    area_basis="solid",
    exclude_border=True,
    histogram_bins=30,
))

# Separate binary route: measure saved masks from BOTH segmentation methods.
# No contours or level sets are used in these measurements.
print(particle_size_distribution(
    images=input_folder / "segmented",
    input_metadata_csv=input_folder / "preprocessing_metadata.csv",
    percentile_range=(10, 90),
    area_basis="solid",
    exclude_border=True,
    histogram_bins=30,
))
```

Use `methods="OTSU"` or `methods="CHAN_VESE"` in
`particle_size_distribution` to restrict the binary route. By default all
successful saved configurations are considered, including earlier settings.
`measure_particles=False` disables the Chan–Vese contour measurement step.

## Results

| File under `segmented/` | Contents |
| --- | --- |
| `contour_particles.csv` | Per-object measurements from Chan–Vese zero contours only |
| `contour_psd_summary.csv` | One contour distribution per image/configuration/analysis recipe |
| `contour_measurements/` | Per-record tables, contour-coordinate NPZs, labelled TIFFs, and plots |
| `binary_descriptors/binary_particles.csv` | Per-object pixel-count measurements from saved masks |
| `binary_descriptors/binary_psd_summary.csv` | One binary distribution per saved mask/analysis recipe |
| `binary_descriptors/*.png` | Binary distribution plots |
| `segmentation_history.csv` | Segmentation rows with separate `contour_psd_*` and `binary_psd_*` statistics |

Each distribution plot contains a count histogram and empirical cumulative
distribution. Statistics include count, mean, median, sample standard deviation,
minimum, maximum, P10/P50/P90 and configurable lower/upper percentiles.
`size_unit` identifies micrometres (`um`) or equivalent pixel units (`px`).

Distribution tables retain every analysis recipe. The per-segmentation history
row shows the latest analysis for each route; the route's summary table retains
earlier percentiles, area choices, calibration and other parameter variants.
Repeated identical calls reuse verified outputs and do not add duplicate rows.
Plots and CSVs are separate for each image and segmentation recipe: results are
not pooled across images or mixed between the contour and binary routes.

## Definitions and interpretation

- Size is the **2D equivalent circular diameter**, `sqrt(4 * area / pi)`.
- Distributions and percentiles are **number-weighted**: each eligible object
  counts once. These are not volume-weighted Dv10/Dv50/Dv90 or 3D particle sizes.
- Percentiles use linear interpolation. `percentile_range=(5, 95)` reports P5
  and P95; it does **not** clip or remove the tails. P10/P50/P90 remain available.
- Standard deviation uses `ddof=1` and is blank when fewer than two objects are
  eligible. Empty distributions have blank size statistics and an explicit plot.
- `area_basis="solid"` excludes internal holes. `"envelope"` includes holes
  inside the outer boundary. The same choice is available in both routes.
- Pixel calibration is matched to each image's CSV row using
  `pixel_size_x_nm` and `pixel_size_y_nm`. Physical area is
  `area_px2 * x_nm * y_nm / 1e6`. No square-pixel assumption is made.
- Missing/invalid calibration leaves physical areas blank and produces a
  distribution explicitly labelled in equivalent pixel units, plus a warning.
- Objects use 4-connectivity. Touching particles remain one labelled region.
  No automatic watershed, smoothing or size filtering is added.
- Border-cut objects remain in particle tables and are excluded from PSDs by
  default. `exclude_border=False` can include partial binary objects. Open
  Chan–Vese contours have no valid enclosed area and remain excluded.
- IDs in particle tables match the labelled TIFFs. Retained contour coordinates
  use original-image `(row, column)` pixel-centre coordinates.
- Original images and masks are not modified. Legacy mixed measurement exports
  remain on disk but are no longer updated. An old Chan–Vese result without a
  level set must be rerun once for continuous-contour measurements.

Close CSV files in Excel before running. Check returned summaries for per-image
errors and calibration warnings. Validate boundaries and particle separation on
representative SEM images before interpreting the measured sizes.

Audit/cache JSON is stored in `json/` subfolders, including
`segmented/json/segmentation_records.json` and a `json/` folder inside each
measurement directory. Existing generated JSON records are migrated on use;
their cached outputs remain reusable. Keep `json/` with its parent output folder.
