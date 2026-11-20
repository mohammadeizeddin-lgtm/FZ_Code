# SEM batch workflow — version 2

Processes one TIFF, a list of TIFFs, or a folder into cropped images, audit JSON
and an incremental metadata CSV. CSV opens in Excel. There is no XLSX workbook.
Run on original images before intensity normalisation or clipping.

## Install and run

Keep all four `sem_*.py` modules together. Use Python 3.11+:

```sh
pip install -r requirements.txt
```

```python
from sem_batch import process_sem_images

summary = process_sem_images(
    images=r"C:\SEM\raw",
    output_folder=r"C:\SEM\processed",
    recursive=True,
)
print(summary)
```

`images` also accepts a single TIFF path or a list of file/folder paths.

Command line:

```sh
python sem_batch.py "C:\SEM\raw" --output "C:\SEM\processed" --recursive
```

## Manufacturer-independent architecture

1. `sem_metadata.py`: read TIFF tags, identify the manufacturer from available
   Make/Model/Software hints, select a metadata adapter, return common fields and
   original namespaced fields.
2. `sem_processing.py`: independently decide whether a boundary is sufficiently
   supported, save a review copy when it is not, or save a user-provided mask.
3. `sem_information_bar.py`: footer-image heuristic and lossless TIFF crop writer.
4. `sem_batch.py`: incremental CSV updates, version/policy/content hashes and CLI.

Manufacturer recognition is not the same as a validated proprietary reader:

| Input metadata | Current handling |
| --- | --- |
| TESCAN tag 50431 | Dedicated binary/text decoder; validated on supplied real image |
| Other/unknown manufacturer's TIFF | Generic TIFF reader plus readable JSON, key/value, INI-like or XML text extraction |
| ZEISS/FEI/JEOL/Hitachi name in Make/Model/Software | Identifies manufacturer when possible; uses generic reader unless a dedicated adapter is registered |
| Undecoded private binary metadata | Archived in the original TIFF and Base64-encoded audit; not presented as decoded instrument fields |

Generic text reading works on the text actually present. It does not claim to
decode all manufacturers' private TIFF formats. Real files from each additional
machine are needed before implementing/validating its dedicated reader.

## Common metadata and units

The CSV includes manufacturer, model, detector, operator, acquisition time/date,
pixel size, accelerating voltage, working distance, dwell time, magnification,
metadata reader/status, calibration status, crop method and review flags.

- TESCAN uses the validated source-unit definitions from its adapter.
- Generic converted fields require an explicit unit in the key, value or a
  companion unit field. Examples: `PixelSizeX_nm=12.5`, `PixelSizeX=12.5 nm`, or
  `PixelSizeX=12.5` plus `PixelSizeXUnit=nm`. Axes are handled independently.
- Explicit XML `PhysicalSizeX/Y` and their unit attributes can be read, but this
  does not make an OME/other structured TIFF safe to crop with a generic writer.
- Unitless or non-positive pixel sizes stay blank; units are never guessed.
  Conflicting explicit calibration values are flagged and not silently selected.
- TIFF print resolution/DPI and magnification are not used to infer specimen
  calibration. The original TIFF resolution fields are still exported.
- Raw instrument strings remain under `tescan.*` or `generic.tag_<code>.*`.
  Readable original TIFF tags are under `tiff.*`. Original binary tags and storage
  arrays are preserved in audit JSON, not expanded into spreadsheet cells.
- Missing values remain blank rather than zero. `calibration_status` reports
  available, partial, unavailable or conflicting. Check it before physical-size
  measurements; successful cropping does not guarantee calibrated measurements.

`metadata_status=vendor_decoded` means the dedicated reader succeeded;
`generic_partial` means the generic reader was used, and private fields may still
be undecoded. `vendor_decode_failed` explicitly records a dedicated-reader failure.

## Crop decisions

Default behaviour is conservative:

- A metadata footer height agreeing with the image boundary is cropped, with
  `crop_method=metadata_and_image`.
- Metadata explicitly indicating zero footer produces an unchanged, byte-exact
  copy with `crop_method=metadata_no_footer`.
- A candidate based only on image structure is NOT cropped by default. It
  produces a byte-identical TIFF under `review/`, an audit, a preview showing a
  proposed boundary (when available), and `status=needs_review` in the CSV.
- Ambiguous/missing boundaries or unsupported export geometry also go to review.
  A missing boundary does not prove an image is free of annotations.

Inspect a review preview and approve a boundary using its zero-based first row
to remove. For example, row 768 retains rows 0 through 767:

```python
summary = process_sem_images(
    r"C:\SEM\raw", r"C:\SEM\processed",
    crop_rows={"example.tif": 768},
)
```

Use a full source path or the CSV `source_id` when filenames are duplicated across
folders. A row equal to the original height explicitly approves an unchanged
image. The existing CSV row is updated, and the previous review copy is retained.
Approved crop rows apply to ORIGINAL source files, not prior derivatives.

An explicit opt-in is available after validating a recurring footer layout:

```python
process_sem_images(raw_folder, output_folder, allow_image_only_crop=True)
```

These outputs remain marked `crop_method=image_structure_unconfirmed` and
`review_required=True`; the option does not turn a heuristic into ground truth.

CLI approval: `--crop-rows decisions.json`, where the JSON maps filenames/full
paths/source IDs to approved row numbers. `--allow-image-only-crop` is the CLI
equivalent of the opt-in above.

## Text/scale bars over the specimen: exclusion masks

The workflow does not automatically locate arbitrary overlays or reconstruct
specimen pixels behind text. Supply an analysis mask after identifying an overlay:

```python
import numpy as np
from sem_batch import process_sem_images

valid = np.ones(loaded_image.shape, dtype=bool)
valid[overlay_top:overlay_bottom, overlay_left:overlay_right] = False

process_sem_images(
    [source_path], output_folder,
    exclusion_masks={source_path: valid},
)
```

True/nonzero means INCLUDE in analysis; False/zero means EXCLUDE. An entry may
also be a grayscale mask image path containing 0/1 or 0/255. Shapes must equal the
original image shape, and at least one pixel must remain included.

Mask-only processing saves a byte-identical source TIFF and a separate
`.analysis_mask.tif` (255 include, 0 exclude), with `crop_method=analysis_mask`.
It does not erase annotated pixels or resize the original. Apply this mask when
calculating percentiles, histograms, thresholds and area denominators; merely
saving the mask does not make later analysis automatically honour it. Do not pass
both a crop decision and a mask for the same source.

## Output and incremental CSV

- `metadata.csv`: one record per resolved source path, including failures/review.
- `images/`: successful cropped or unchanged/mask-only TIFFs, audit JSON and masks.
- `review/`: unchanged original TIFF copies, review previews and audit JSON.

Run again after adding files. New paths add rows, changed inputs update their
existing rows, and unchanged verified inputs are skipped without CSV rewrites.
Review rows also skip if nothing changed; adding an approval/mask changes the
policy hash and updates the same row. Changes in script/reader versions or the
processing policy invalidate skip decisions. Earlier output versions are retained.
Lost CSVs can be rebuilt from raw sources and verified audit/output pairs.

CSV source identity is the resolved path, not filename alone. Moving/renaming a
source creates a new identity; identical files at different paths remain separate
records. Duplicate arguments resolving to the same path are processed once.
Removed inputs do not silently delete historical CSV rows. Missing/damaged
outputs are regenerated into an available filename.

Existing version-1 CSVs are accepted and upgraded when their sources are processed:
new columns are added without duplicating source rows or discarding custom notes.
Rows for sources not included in that run remain unchanged. Source identity/status,
common managed fields and `tiff.*`/`tescan.*`/`generic.*` columns belong to the script;
use distinct custom columns such as `review_notes` for manual entries.

Close the CSV in Excel while running. UTF-8 with BOM is used for Excel text
compatibility. Complete CSV updates use atomic replacement and detect concurrent
external changes. A lock prevents concurrent batch runs on one output folder.
After a crash, remove `.sem_batch.lock` only once no process is still using it.
Unrecognised or malformed existing CSVs are refused rather than overwritten.

Summary `added`/`updated` counts include new/updated review records. `needs_review`
counts newly handled review cases; `review_required` also includes unchanged
review rows and explicitly unconfirmed opt-in crops. `failed` counts per-file
errors. Read `status`, `review_required`, `metadata_status`, `calibration_status`
and error/review-reason columns before downstream analysis. The CLI exits 1 if
any failures occurred; review cases are reported in its JSON summary.

## Preservation and support limits

Raw source files are never modified. Retained cropped pixels and dtype are
verified after reloading, and compression/descriptive tags are checked. Cropping
changes dimensions and regenerates TIFF storage offsets/counts. The output file
is not byte-identical to the source after a crop. TESCAN ImageStripSize is updated
to 0; other known generic JSON/text footer or shape fields are updated where
supported. Full original tags are always archived for provenance.

Unknown vendor binary blocks are archival copies, not a guarantee that proprietary
software will interpret every internal reference as a current derivative. Original
acquisition thumbnails can still show the original footer. Original source files
remain the reference for full metadata and the printed scale bar.

The crop exporter currently supports classic little-endian, single-page,
standard-orientation, black-is-zero, uint8/uint16 grayscale TIFFs with supported
lossless compression. Crops of big-endian/BigTIFF, stacks, linked IFD metadata, RGB images,
and structured XML geometry requiring a dedicated exporter are preserved for
review, rather than silently converted. Non-TIFF file formats are not yet batch
inputs. Known-footer detection still assumes a separate flat-background bottom
strip; it is not a universal text/scale-bar detector.

## Validation and extending readers

`VALIDATION.json` records the checks. The supplied real TESCAN TIFF was verified:
1024×887 to 1024×768, uint16 and compression preserved, 119 rows removed,
9.540838 nm/pixel in both axes, exact retained pixel equality, unchanged source.

Other manufacturer names, explicit/missing/conflicting units, generic footer
metadata, review approval, XML preservation and masks were checked using clearly
synthetic TIFF fixtures. They are not evidence that all ZEISS/FEI/JEOL/Hitachi
exports are supported. Additional real files are needed to validate those formats.
The example output contains ONLY the real TESCAN sample, not fabricated datasets.
Start with a fresh output folder for your own local files: the example records
the original processing environment's absolute source path.

Add dedicated metadata readers through
`register_metadata_reader(name, predicate, reader)` in `sem_metadata.py`. Readers
receive raw TIFF tags and a Metadata object. They should expose original fields
under a unique prefix, populate common fields only with verified units, report
calibration/status, and provide an unambiguous footer height when known. Use a new
versioned reader name when its interpretation changes. Crop tag-update rules and
the export engine also need validation if a vendor format embeds geometry/offsets.

Reference documentation: Pillow TIFF `tag_v2`/`tiffinfo` at
https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html and OME
physical-size/unit attributes at
https://www.openmicroscopy.org/Schemas/Documentation/Generated/OME-2016-06/ome_xsd.html.

# Batch preprocessing of SEM image outputs

`sem_preprocess.py` contains your existing names: `parse_path`, `load_tiff`,
`preprocess_image` and `save_tiff`. The new batch function is `preprocess_images`.
Existing flags remain `clipping_percentile_range`, `gaussian_correction_filter`,
`background_sigma` and `gentle_denoising`. All optional processing defaults off;
min/max scaling to 0–1 is always performed on successful images.

## Install

Use the six project dependencies in `requirements.txt` (current stable pins
checked 13 November 2026). These pins require Python 3.12 or newer:

```sh
python -m pip install -r requirements.txt
```

## Your extraction-to-preprocessing workflow

Leave the extraction step and its CSV unchanged. Then run:

```python
from pathlib import Path
from sem_preprocess import preprocess_images

input_folder = Path(
    r"C:\Users\f842198\OneDrive - Fortescue Metals Group\Desktop\Moha\Moha\2026\11_26\LFP_S2C\processed"
)

summary = preprocess_images(
    images=input_folder,
    output_folder=input_folder / "preprocessed",
    clipping_percentile_range=None,
    gaussian_correction_filter=False,
    background_sigma=None,
    gentle_denoising=False,
    recursive=False,
)
print(summary)
```

Enable options individually when needed. Example only, requiring feature checks:

```python
summary = preprocess_images(
    images=input_folder,
    output_folder=input_folder / "preprocessed",
    clipping_percentile_range=(1, 99),
    gaussian_correction_filter=True,
    background_sigma=100,
    gentle_denoising=True,
)
```

The sigma of 100 pixels and NLM h factor 0.6 are starting examples, not validated
settings for every SEM image. NLM can mistake fine texture for noise; background
subtraction can remove genuine broad material gradients. Validate pores, cracks,
particle boundaries and texture before extracting measurements.

## Input selection

- One image, a list of images/folders, or an input folder is accepted.
- When the extraction `metadata.csv` is present, only active `status=success`
  outputs with no review flag are selected. Its `output_image` paths identify
  the images even if they are inside `images/`, so `recursive=False` works when
  you pass the extraction output root. Old image versions, review copies and
  analysis-mask TIFFs are not ingested as specimen images.
- Passing the extraction `images/` folder or an individual active image also
  finds the standard parent metadata CSV.
- A nonstandard extraction CSV path can be supplied as `input_metadata_csv=...`.
- Without an extraction CSV, `recursive=False` scans the given folder directly;
  True includes subfolders. The selected output folder and files ending in
  `.analysis_mask.tif` / `.mask.tif` are excluded from plain-folder discovery.
- A recognised extraction CSV must have `source_id`, `status`, `output_image`
  columns. An unrecognised `metadata.csv` is refused rather than guessed.
- Images requiring an exclusion mask are logged as `needs_review`. The supplied
  `preprocess_image` is not mask-aware and must not filter across invalid overlays.
  Implement mask-aware preprocessing before enabling those cases.

The batch does not rely on your filenames matching `parse_path`'s convention.
Use that helper separately when filenames follow detector_state_chemistry_ID.

## Saved results and CSV history

The output folder contains:

- Derived float32 TIFFs with the same image dimensions and values in 0–1.
- `preprocessing_metadata.csv`, separate from the original extraction CSV.
- One audit JSON per image, containing original TIFF metadata, preprocessing
  details, upstream metadata, hashes, versions and verification results.

The extended CSV retains the existing acquisition metadata and appends only:

| New column | Meaning |
| --- | --- |
| `image_file` | Input image filename |
| `processed_image` | Saved processed TIFF filename |
| `clipping` | `True` when a percentile range was supplied, otherwise `False` |
| `clipping_percentile_range` | Supplied range, e.g. `(1, 99)`, or literal `None` |
| `gaussian_correction_filter` | `True` or `False` |
| `background_sigma` | Supplied sigma when correction is enabled; literal `None` when disabled |
| `gentle_denoising` | `True` or `False` |

No noise estimates, runtime statistics, intensity limits, hashes or timestamps
are added as CSV columns. For example:

```csv
clipping,clipping_percentile_range,gaussian_correction_filter,background_sigma,gentle_denoising
False,None,False,None,False
True,"(1, 99)",True,100,False
```

Disabled Gaussian correction always logs `None` sigma, including when an unused
sigma argument was supplied. That inactive argument does not change the recipe.
A supplied `(0, 100)` range is logged as clipping enabled, although it changes no
extreme values. These columns record the selected parameters, not noise estimates.

Only successfully saved image records are included in the parameter CSV. Failed
or mask-review cases are retained in the returned summary and the JSON history,
so they are not presented as successfully processed images.

Detailed diagnostics and bookkeeping are retained outside the CSV, in per-image
audit JSON and `preprocessing_metadata.records.json` (or the corresponding name
for a custom CSV). This preserves incremental caching, error records and earlier
history without cluttering the spreadsheet. Keep these JSON files with the TIFFs.
Existing detailed CSVs from the earlier version are automatically migrated: their
full records are retained in JSON, and their public CSV columns are simplified.
Existing TIFFs are not reprocessed merely to change the log layout.

An identical source path, image content, settings/implementation/runtime and
upstream metadata snapshot has one processing record. Rerunning unchanged inputs
skips verified outputs and does not rewrite an unchanged CSV. New images append
rows. Different settings or changed input content/metadata append a new history
record and preserve earlier outputs. Adding empty upstream columns for older
images does not force reprocessing.

Missing/damaged outputs are regenerated into an available filename. File-specific
failures are recorded and the batch continues. There is no silent fallback that
turns requested denoising off if a dependency is missing. Failed cases are retried
on reruns, rather than duplicated under the same record ID.

User-added notes are retained. Acquisition and parameter columns are managed by
the workflow; use a separate column ending in `_notes` for editable review notes. Existing logs
must use the current or earlier workflow schema. Close the CSV in Excel during processing.
Updates are atomic and detect concurrent external edits. A lock prevents two
batch instances writing the same folder; remove a stale `.preprocessing.lock`
only after confirming that the earlier process has stopped.

## Preservation and I/O

Input images and the original extraction CSV are never modified. The output is
a DERIVED float32 TIFF, not the original integer acquisition format. Reloaded
output dtype and pixels are checked exactly against the processed array.

Calibration fields from the extraction CSV/audit are carried into the extended
CSV and the derived TIFF's JSON description. Existing standard TIFF resolution
tags are copied where present; they are not inferred to be specimen calibration.
All original TIFF tags, including binary vendor blocks, are archived in JSON
(binary values Base64-encoded). Original proprietary blocks are not blindly
embedded in a TIFF with a changed intensity representation; keep source TIFFs
for full original vendor compatibility and acquisition metadata.

`tifffile` is the preferred TIFF reader/writer. If unavailable, Pillow is an
explicitly logged fallback. Single-page standard-orientation black-is-zero
grayscale images are supported. Stacks, unsupported orientations, RGB and
nonfinite/constant arrays are rejected rather than silently transformed.
Some compressed TIFF codecs may require additional decoder support; failures
are logged. No resizing or resampling is performed.

## Standalone function compatibility

```python
processed_image = preprocess_image(image, gentle_denoising=True)
processed_image, details = preprocess_image(image, return_details=True)
save_tiff(processed_image, output_path)
```

The normal return value remains a float32 array; the optional `return_details`
tuple is used internally by the batch to log actual intensity/noise limits.
The denoising calculation matches the last function you provided: NLM on raw
floating-point intensities, h=0.6*estimated_sigma, patch_size=5, patch_distance=6,
fast_mode=True, preserve_range=True.

Command line:

```sh
python sem_preprocess.py "C:\SEM\processed" --output "C:\SEM\processed\preprocessed"
```

Optional CLI flags: `--recursive`, `--clip 1 99`, `--gentle-denoising`,
`--gaussian-correction-filter --background-sigma 100`, `--input-metadata-csv`.

## Included demonstration and validation limits

`example_output/` contains the supplied real TESCAN image after prior footer
removal, processed with (1) normalisation only and (2) Gaussian correction at
sigma 100 plus 1–99 clipping. Both are illustrative; this does not establish
that background correction improves the scientific segmentation of this image.
The example CSV has two history rows: 143 acquisition columns plus the seven simple filename/parameter columns (150 total).
Start a fresh output folder for your own local run; example paths refer to the
environment where the demonstration was generated.

`VALIDATION.json` records the completed batch lifecycle checks and limits.
Exact TIFF roundtrip, default/background/clipping paths and CSV lifecycle were
tested using Pillow fallback because native tifffile and scikit-image were not
installed in this execution environment. Native tifffile/NLM execution still
requires local verification with your installed dependencies. The failure path
for a missing NLM dependency was verified, with no falsely labelled output.

# SEM particle-size distributions

Keep `segmentation.py` and `extracting_discriptors.py` together. The second
filename preserves the spelling requested. Dependencies: NumPy, SciPy,
scikit-image, Pillow and Matplotlib.

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
