# Batch preprocessing of SEM image outputs

`sem_preprocess.py` contains your existing names: `parse_path`, `load_tiff`,
`preprocess_image` and `save_tiff`. The new batch function is `preprocess_images`.
Existing flags remain `clipping_percentile_range`, `gaussian_correction_filter`,
`background_sigma` and `gentle_denoising`. All optional processing defaults off;
min/max scaling to 0–1 is always performed on successful images.

## Install

Use the seven project dependencies in `requirements.txt` (current stable pins
checked 13 November 2026). These pins require Python 3.12 or newer:

```sh
python -m pip install -r requirements.txt
```

Gentle denoising uses `estimate_sigma`, which additionally requires PyWavelets
(imported as `pywt`). It is now explicitly pinned in requirements.txt. If your
existing environment reports a missing PyWavelets dependency, install
`PyWavelets==1.10.0` using the same Python interpreter that runs the script.


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
