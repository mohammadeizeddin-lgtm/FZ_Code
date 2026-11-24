# Progress for the SEM workflow

Keep `sem_progress.py` and `sem_paths.py` beside the other Python modules. They use only Python's
standard library, so there is no extra package to install.

Percentage progress is enabled by default for all four batch functions:

```python
from sem_extract_info import process_sem_images
from preprocessing import preprocess_images
from segmentation import segment_images
from extracting_discriptors import particle_size_distribution

summary = process_sem_images(images=raw_folder, output_folder=processed_folder,
                             show_progress=True)
summary = preprocess_images(images=processed_folder, output_folder=preprocessed_folder,
                            show_progress=True)
summary = segment_images(images=preprocessed_folder, method="CHAN_VESE",
                         show_progress=True)
summary = particle_size_distribution(images=preprocessed_folder / "segmented",
                                     show_progress=True)
```

`sem_batch.py` and `sem_preprocess.py` remain the implementation modules.
`sem_extract_info.py` and `preprocessing.py` preserve the import names used in
your previous examples. Keep all files together when using these aliases.

Use `show_progress=False` to disable progress. Command-line entry points accept
`--no-progress`. Existing parameter names, defaults and returned summaries are
preserved. Single-image helpers such as `preprocess_image`, `load_tiff`,
`save_tiff`, `otsu_thresholding`, `chan_vese_segmentation`,
`remove_information_bar` and `extract_sem_metadata` also accept
`show_progress=False` (and display progress by default).

## What the percentage means

- Batch progress counts inspected images/segmentation records. Successful,
  cached, failed and review-required items all count towards completion.
- A percentage of 100 means all batch items have been visited; the display can
  briefly show the final CSV-saving stage before the function returns.
- A long native operation, such as Chan–Vese or non-local means, shows its active
  function and elapsed time. Its percentage does not advance until an image
  finishes. No iterative solver is restarted or changed to invent a percentage.
- A directly called single-image function shows 0% while running and 100% when
  complete. A terminal refreshes elapsed time while a native call blocks.
- Nested functions update the batch stage rather than creating overlapping bars.
- Output goes to stderr, keeping command-line JSON on stdout usable. Existing
  informational stdout messages, such as the preprocessing shape/dtype line,
  remain. Piped output and notebooks use throttled plain-text updates.
- An unhandled exception closes the display as stopped; it does not force a
  failed operation to 100%. Display/output-stream errors do not interrupt work.

The display does not change scientific settings, calculations, source images,
CSV schemas or output recipes. Function-source hashes used by existing caches
are preserved, so verified prior outputs are still skipped.

The separate contour and binary particle-size routes are documented in
`README_particle_size_distributions.md`. Their measurement behaviour is retained.

Generated audit/cache JSON now lives in `json/` subfolders. Existing records are
migrated when their processing stage runs. To use automatic preprocessing folder
names, omit `output_folder` and take the next stage's input from
`Path(summary["output_folder"])`: the folder is named `<sample>_preprocessed`,
with `_clipped` appended when a clipping range is supplied. Existing explicit
paths continue to work; image filenames are unchanged.
