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
