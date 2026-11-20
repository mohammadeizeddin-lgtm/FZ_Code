r"""Batch Otsu or Chan-Vese segmentation of preprocessing outputs.

Usage:
    from segmentation import segment_images
    summary = segment_images(images=preprocessed_folder, foreground="bright")

Outputs default to preprocessed_folder/segmented. The existing preprocessing CSV
gets only segmented (yes/no) and method_of_segmentation (OTSU/CHAN_VESE/None).
The method column accumulates successful methods, e.g. OTSU; CHAN_VESE.
segmented/segmentation_history.csv retains one row per image/configuration,
including parameters and output filenames. Existing JSON history is imported
automatically, so earlier methods can be recovered without rerunning them.
Original images and existing acquisition/preprocessing column values are retained.

Example:
    from pathlib import Path
    from segmentation import segment_images
    input_folder = Path(r"C:\SEM\processed\preprocessed")
    summary = segment_images(images=input_folder,
        output_folder=input_folder / "segmented", method="OTSU",
        foreground="bright", nbins=256)
    print(summary)

Select Chan-Vese by setting method="CHAN_VESE" instead of method="OTSU".
Optional tuning: chan_vese_parameters={"mu": 0.1, "max_num_iter": 500}.
Unspecified parameters use the documented defaults in CHAN_VESE_DEFAULTS.

Masks: uint8 TIFF, 0=background and 255=selected foreground, original dimensions.
Overlays: lossless RGB PNG with foreground tinted red; visualisation only.
The CSV is Excel-readable; no separate XLSX workbook is generated.
Detailed diagnostics stay in segmented/segmentation_records.json.
Keep extracting_discriptors.py beside this file. This module measures ONLY
retained Chan-Vese contours when measure_particles=True (default). Otsu does
segmentation only here: no contour or pixel-count measurements are made for it.
Binary-mask measurements for EITHER method are a separate call:
    from extracting_discriptors import particle_size_distribution
    particle_size_distribution(images=input_folder / 'segmented')
Contour outputs: segmented/contour_particles.csv, contour_psd_summary.csv and
per-record tables, particle-label TIFFs, contour NPZs and distribution plots in
segmented/contour_measurements/. Particle IDs match the labelled TIFF values.
Chan-Vese retains a foreground-positive floating-point *_level_set.npy and uses
its zero contour. The binary descriptor route counts foreground pixels only.
Both routes make histogram/empirical cumulative plots and report number-weighted
2D equivalent circular diameter statistics; default percentiles are P10/P90.
percentile_range changes reported limits, not clipping or particle selection.
segmentation_history.csv contains separate contour_psd_* and binary_psd_* summary
columns. Each route's psd_summary.csv retains all parameter/calibration variants.
Legacy mixed particle_measurements.csv/measurements outputs are left unchanged.
For an older Chan-Vese result, rerun once to obtain its previously unsaved level
set; earlier segmentation/measurement histories are retained.
Calibration uses CSV pixel_size_x_nm and pixel_size_y_nm for each image; areas
are converted using x_nm*y_nm/1e6. No assumed pixel size or unit inference.
Missing/invalid calibration leaves physical values blank and generates a warning.
Its distribution is explicitly labelled in equivalent pixel units instead.
Objects touching the image edge are retained/flagged, with contour area blank.
Solid areas exclude holes; envelope areas include them. Touching particles remain
one connected region. These are 2D measurements and require boundary validation.
Close the CSV in Excel during updates. Run via a separate driver script or supply
the input folder when invoking this module directly from the command line.
"""
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import argparse
import csv
import hashlib
import inspect
import json
import os
import tempfile

import numpy as np
from PIL import Image, TiffImagePlugin
try:
    from skimage import io
    from skimage.filters import threshold_otsu
except ImportError:
    # Keep batch helpers importable; never silently substitute another algorithm.
    io = None
    threshold_otsu = None

try:
    from skimage.segmentation import chan_vese
except ImportError:
    chan_vese = None
try:
    from skimage.measure import find_contours
except ImportError:
    find_contours = None

VERSION = '1.1.0'
SEGMENTATION_COLUMNS = ['segmented', 'method_of_segmentation']
CHAN_VESE_DEFAULTS = dict(mu=0.25, lambda1=1.0, lambda2=1.0, tol=1e-3,
                          max_num_iter=500, dt=0.5, init_level_set='checkerboard')


def _chan_vese_parameters(parameters):
    """Validate and expand a JSON-serialisable subset of scikit-image options."""
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError('chan_vese_parameters must be a dictionary or None.')
    parameters = {} if parameters is None else parameters
    unknown = set(parameters) - set(CHAN_VESE_DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown Chan-Vese parameters: {sorted(unknown)}.')
    settings = {**CHAN_VESE_DEFAULTS, **parameters}
    for name in ('mu', 'lambda1', 'lambda2', 'tol', 'dt'):
        value = settings[name]
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value < 0
                or (name in ('lambda1', 'lambda2', 'dt') and value == 0)):
            raise ValueError(f'Chan-Vese {name} must be finite and '
                             + ('positive.' if name in ('lambda1', 'lambda2', 'dt') else 'non-negative.'))
        settings[name] = float(value)
    iterations = settings['max_num_iter']
    if (isinstance(iterations, (bool, np.bool_))
            or not isinstance(iterations, (int, np.integer)) or iterations < 1):
        raise ValueError('Chan-Vese max_num_iter must be a positive integer.')
    settings['max_num_iter'] = int(iterations)
    if (not isinstance(settings['init_level_set'], str)
            or settings['init_level_set'] not in ('checkerboard', 'disk', 'small disk')):
        raise ValueError("init_level_set must be 'checkerboard', 'disk', or 'small disk'.")
    return settings


def chan_vese_segmentation(image, foreground='bright', *,
                           chan_vese_parameters=None, return_details=False,
                           return_level_set=False):
    """Return a Boolean mask using scikit-image's iterative Chan-Vese method.

    The two resulting regions are labelled by their mean input intensity so
    foreground='bright'/'dark' has the same meaning as for Otsu. This relabelling
    does not impose a global threshold. lambda1/lambda2 refer to the solver's
    original regions, before relabelling; equal weights are the default.

    Input is copied to float64 and linearly scaled to [0, 1] without clipping.
    No smoothing, downsampling, hole filling or object removal is added here.
    mu controls boundary regularisation; increasing it can remove small features.
    Initialisation and iteration limits can affect the result. Inspect overlays.
    Empty/full or equal-mean partitions are rejected as ambiguous foregrounds.
    return_details=True returns (mask, diagnostics), including iteration energies.
    return_level_set=True returns (mask, diagnostics, signed_level_set), with
    positive values inside the selected foreground, suitable for zero contours.
    """
    image = np.asarray(image)
    if image.ndim != 2 or image.size == 0:
        raise ValueError('Input image must be a non-empty 2D array.')
    if np.iscomplexobj(image) or not np.isfinite(image).all():
        raise ValueError('Input image must contain finite real intensities.')
    if foreground not in ('dark', 'bright'):
        raise ValueError("Foreground must be either 'dark' or 'bright'.")
    settings = _chan_vese_parameters(chan_vese_parameters)
    values = image.astype(np.float64)
    low, high = float(values.min()), float(values.max())
    if high <= low or not np.isfinite(high - low):
        raise ValueError('Image has no usable intensity variation for Chan-Vese.')
    values = (values - low) / (high - low)
    if chan_vese is None:
        raise ImportError('Chan-Vese requires scikit-image: python -m pip install scikit-image')
    mask, level_set, energies = chan_vese(values, **settings, extended_output=True)
    mask = np.asarray(mask, dtype=bool)
    energies = np.asarray(energies, dtype=float)
    if (mask.shape != image.shape or not np.isfinite(level_set).all()
            or not np.isfinite(energies).all()):
        raise ValueError('Chan-Vese returned an invalid or non-finite result.')
    if not mask.any() or mask.all():
        raise ValueError('Chan-Vese did not separate two regions; review its parameters.')
    inside_mean = float(values[mask].mean())
    outside_mean = float(values[~mask].mean())
    if np.isclose(inside_mean, outside_mean, rtol=1e-7, atol=1e-12):
        raise ValueError('Chan-Vese region means are indistinguishable; cannot label bright/dark.')
    invert = (inside_mean > outside_mean) != (foreground == 'bright')
    if invert:
        mask = ~mask
    selected_level_set = np.asarray(level_set, dtype=np.float64).copy()
    if invert:
        selected_level_set *= -1
    # Exact zero samples follow the chosen foreground (dark includes ties).
    zeros = selected_level_set == 0
    epsilon = np.finfo(np.float64).eps * max(1., float(np.abs(selected_level_set).max()))
    selected_level_set[zeros] = np.where(mask[zeros], epsilon, -epsilon)
    if not np.array_equal(selected_level_set > 0, mask):
        raise ValueError('Chan-Vese mask and level-set signs disagree.')
    details = dict(input_scaling='linear_min_max_0_1_no_clipping',
        scaling_low=low, scaling_high=high, iterations=int(energies.size),
        iteration_limit_reached=bool(energies.size >= settings['max_num_iter']),
        energies=energies.tolist(), foreground_relabelled=bool(invert),
        solver_true_region_mean=inside_mean, solver_false_region_mean=outside_mean,
        level_set_zero_samples_adjusted=int(zeros.sum()))
    # Reaching the limit is recorded; it is not a claim about convergence.
    if return_level_set:
        return mask, details, selected_level_set
    return (mask, details) if return_details else mask


def otsu_thresholding(image, foreground="bright", nbins=256):
    """Return (Boolean mask, float threshold) using your original Otsu method.

    Bright foreground: image > threshold. Dark foreground: image <= threshold.
    nbins controls the histogram for floating-point images; scikit-image ignores
    it for integer arrays. Run on the preprocessed image without extra scaling.
    """
    image = np.asarray(image)
    if image.ndim != 2 or image.size == 0:
        raise ValueError('Input image must be a non-empty 2D array.')
    if np.iscomplexobj(image) or not np.isfinite(image).all():
        raise ValueError('Input image must contain finite real intensities.')
    if image.min() == image.max():
        raise ValueError('Image has no intensity variation; no meaningful Otsu split.')
    if foreground not in ('dark', 'bright'):
        raise ValueError("Foreground must be either 'dark' or 'bright'.")
    if isinstance(nbins, (bool, np.bool_)) or not isinstance(nbins, (int, np.integer)) or nbins < 2:
        raise ValueError('nbins must be an integer of at least 2.')
    if threshold_otsu is None:
        raise ImportError('Otsu segmentation requires scikit-image: python -m pip install scikit-image')
    threshold_value = threshold_otsu(image, nbins=nbins)
    if foreground == 'dark':
        mask = (image <= threshold_value)
    else:
        mask = (image > threshold_value)
    return mask, float(threshold_value)


def _hash_file(file):
    digest = hashlib.sha256()
    with open(file, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _inside(file, folder):
    try:
        file.relative_to(folder)
        return True
    except ValueError:
        return False


def _versions():
    result = {}
    for package in ('numpy', 'Pillow', 'scikit-image'):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = 'not_installed'
    return result


def _atomic_json(file, value, expected_hash=None):
    handle = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=file.parent,
        prefix='.segmentation_', suffix='.tmp', delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, indent=2, ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        current = _hash_file(file) if file.exists() else None
        if current != expected_hash:
            raise RuntimeError(f'File changed during processing: {file.name}. Rerun to merge the latest state.')
        os.replace(temporary, file)
    finally:
        temporary.unlink(missing_ok=True)


def _read_csv(file):
    if not file.is_file():
        raise FileNotFoundError(f'Preprocessing CSV not found: {file}. Point images at the preprocessed folder.')
    digest = _hash_file(file)
    with open(file, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or len(fields) != len(set(fields)):
        raise ValueError('CSV has missing/duplicate column headings; it was not changed.')
    image_column = next((name for name in ('processed_image', 'preprocess.output_image') if name in fields), None)
    if image_column is None:
        raise ValueError('Expected processed_image in the preprocessing CSV; select the preprocessing output folder.')
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError('CSV has malformed rows; it was not changed.')
    return fields, rows, image_column, digest


def _write_csv(file, rows, fields, expected_hash):
    handle = tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8-sig', dir=file.parent,
        prefix='.segmentation_csv_', suffix='.tmp', delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        if (_hash_file(file) if file.exists() else None) != expected_hash:
            raise RuntimeError('Preprocessing CSV changed during segmentation; rerun to merge the latest file.')
        os.replace(temporary, file)
    except PermissionError as exc:
        raise PermissionError('Close the CSV in Excel and rerun segmentation.') from exc
    finally:
        temporary.unlink(missing_ok=True)


def _load_image(file):
    # Previous-stage outputs are single-page, black-is-zero grayscale TIFFs.
    with Image.open(file) as source:
        if source.format != 'TIFF' or source.n_frames != 1:
            raise ValueError('Expected a single-page TIFF.')
        if source.tag_v2.get(274, 1) != 1 or source.tag_v2.get(262, 1) != 1:
            raise ValueError('Unsupported orientation or grayscale interpretation; no silent conversion.')
        resolution = {code: source.tag_v2[code] for code in (282, 283, 296) if code in source.tag_v2}
        image = np.asarray(source).copy()
    return image, resolution


def _make_overlay(image, mask, overlay_alpha):
    """Visualisation only: red tint marks selected foreground, no resampling."""
    values = image.astype(np.float64)
    if values.min() >= 0 and values.max() <= 1:
        gray = values
    else:
        gray = (values - values.min()) / (values.max() - values.min())
    gray = np.rint(np.clip(gray, 0, 1) * 255).astype(np.uint8)
    overlay = np.repeat(gray[..., None], 3, axis=-1)
    overlay[mask] = np.rint((1 - overlay_alpha) * overlay[mask]
                          + overlay_alpha * np.array([255, 0, 0])).astype(np.uint8)
    return overlay


def _valid_outputs(record, folder, source_hash, recipe_hash):
    if (not isinstance(record, dict) or record.get('status') != 'success'
            or record.get('source_sha256') != source_hash or record.get('recipe_sha256') != recipe_hash):
        return False
    keys = ['mask', 'overlay']
    if record.get('settings', {}).get('method') == 'CHAN_VESE':
        keys.append('level_set')
    for key in keys:
        name = record.get(key + '_file', '')
        if not name:
            return False
        file = (folder / name).resolve()
        if not _inside(file, folder) or not file.is_file() or _hash_file(file) != record.get(key + '_sha256'):
            return False
    return True


def _save_outputs(image, mask, threshold_value, source, folder, record_id,
                  method, foreground, nbins, overlay_alpha, resolution, calibration,
                  settings=None):
    # Short prefixes limit path growth on Windows; the hash distinguishes images
    # and settings even when their visible filename prefixes are identical.
    stem = source.stem[:42] + f'__{method}_{foreground}__{record_id[:12]}'
    count = 1
    while True:
        suffix = '' if count == 1 else f'_{count}'
        mask_file = folder / f'{stem}{suffix}_mask.tif'
        overlay_file = folder / f'{stem}{suffix}_overlay.png'
        if not mask_file.exists() and not overlay_file.exists():
            break
        count += 1
    mask_pixels = np.where(mask, 255, 0).astype(np.uint8)
    overlay_pixels = _make_overlay(image, mask, overlay_alpha)
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[270] = json.dumps(dict(source_image=source.name, method=method,
        foreground=foreground, threshold_value=threshold_value, nbins=nbins,
        settings=settings,
        mask_values={'background': 0, 'foreground': 255}, calibration=calibration), ensure_ascii=True)
    for code, value in resolution.items():
        info[code] = value
    created = []
    try:
        with open(mask_file, 'xb') as f:
            created.append(mask_file)
            Image.fromarray(mask_pixels).save(f, format='TIFF', compression='raw', tiffinfo=info)
        with open(overlay_file, 'xb') as f:
            created.append(overlay_file)
            Image.fromarray(overlay_pixels).save(f, format='PNG')
        with Image.open(mask_file) as saved:
            reloaded_mask = np.asarray(saved)
            if reloaded_mask.dtype != np.uint8 or not np.array_equal(reloaded_mask, mask_pixels):
                raise RuntimeError('Saved mask failed exact pixel verification.')
        with Image.open(overlay_file) as saved:
            if not np.array_equal(np.asarray(saved), overlay_pixels):
                raise RuntimeError('Saved overlay failed exact pixel verification.')
    except Exception:
        for file in created:
            file.unlink(missing_ok=True)
        raise
    return mask_file, overlay_file


def _pixel_calibration(metadata_rows):
    """Use unit-explicit CSV fields only; never guess missing axes or units."""
    parsed = []
    for row in metadata_rows:
        values = []
        for name in ('pixel_size_x_nm', 'pixel_size_y_nm'):
            try:
                value = float(row.get(name, ''))
                values.append(value if np.isfinite(value) and value > 0 else None)
            except (ValueError, TypeError):
                values.append(None)
        status = str(row.get('calibration_status', '')).strip().lower()
        parsed.append((*values, status))
    if not parsed:
        parsed = [(None, None, '')]
    x, y, status = parsed[0]
    valid = x is not None and y is not None
    note = '' if valid else 'Missing, non-positive or invalid pixel_size_x_nm/pixel_size_y_nm.'
    if status not in ('', 'available', 'valid', 'validated', 'verified', 'calibrated'):
        valid, note = False, f'CSV calibration_status is {status!r}; physical units withheld.'
    if any(item != parsed[0] for item in parsed[1:]):
        valid, note = False, 'Conflicting calibration in CSV rows for this image.'
    return dict(pixel_size_x_nm=x, pixel_size_y_nm=y, calibration_status=status,
                calibration_valid=valid, calibration_note=note)


def _polygon_area_rc(points):
    """Signed polygon area in pixel squared; translate to reduce cancellation."""
    points = np.asarray(points, dtype=np.float64)
    xy = points[:, ::-1] - points[0, ::-1]
    return float(0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1)
                              - xy[:, 1] * np.roll(xy[:, 0], -1)))


def _particle_regions(mask, calibration, signed_level_set):
    """Measure ONLY Chan-Vese zero contours; no pixel-count area measurements.

    A retained selected-foreground-positive floating-point level set is required.
    No watershed, filtering, object removal, convex hull or smoothing is applied.
    A labelled region may contain touching particles. IDs are row-major labels.
    Contours are saved in original-image (row, column) pixel-centre coordinates.
    """
    from scipy import ndimage
    if find_contours is None:
        raise ImportError('Particle contour measurements require scikit-image.')
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError('Expected a non-empty 2D mask.')
    if signed_level_set is None:
        raise ValueError('Contour measurements require the retained Chan-Vese level set.')
    signed_level_set = np.asarray(signed_level_set, dtype=np.float64)
    if (signed_level_set.shape != mask.shape or not np.isfinite(signed_level_set).all()
            or not np.array_equal(signed_level_set > 0, mask)):
        raise ValueError('Level set must be finite, match the mask, and be positive in foreground.')
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, 1))
    objects = ndimage.find_objects(labels)
    rows, contours = [], {}
    pixel_area_um2 = None
    if calibration['calibration_valid']:
        pixel_area_um2 = calibration['pixel_size_x_nm'] * calibration['pixel_size_y_nm'] / 1e6
        if not np.isfinite(pixel_area_um2) or pixel_area_um2 <= 0:
            raise ValueError('Calibration produces an unusable physical pixel area.')
    for particle_id, box in enumerate(objects, start=1):
        if box is None:
            continue
        r0, r1, c0, c1 = box[0].start, box[0].stop, box[1].start, box[1].stop
        rstart, rstop = max(0, r0 - 1), min(mask.shape[0], r1 + 1)
        cstart, cstop = max(0, c0 - 1), min(mask.shape[1], c1 + 1)
        local_labels = labels[rstart:rstop, cstart:cstop]
        component = local_labels == particle_id
        border = r0 == 0 or c0 == 0 or r1 == mask.shape[0] or c1 == mask.shape[1]
        field = signed_level_set[rstart:rstop, cstart:cstop].copy()
        # Isolate this object's zero loops; preserve its continuous crossings.
        field[~component] = -np.abs(field[~component])
        loops = find_contours(field, level=0.0, fully_connected='low', positive_orientation='high')
        closed = bool(loops) and all(len(loop) >= 4 and np.array_equal(loop[0], loop[-1]) for loop in loops)
        areas = [_polygon_area_rc(loop) for loop in loops] if closed else []
        net_area, envelope_area, holes = None, None, None
        status = 'border_truncated' if border else 'contour_not_closed'
        if not border and closed:
            outer = int(np.argmax(np.abs(areas)))
            sign = np.sign(areas[outer])
            # One outer loop, all remaining loops must be oppositely oriented holes.
            if sign and sum(np.sign(a) == sign for a in areas) == 1:
                envelope_area = abs(areas[outer])
                net_area = float(sum(areas) * sign)
                holes = len(areas) - 1
                status = 'ok' if 0 < net_area <= envelope_area else 'contour_topology_invalid'
                if status != 'ok':
                    net_area, envelope_area = None, None
            else:
                status = 'contour_topology_invalid'
        for index, loop in enumerate(loops, start=1):
            contours[f'particle_{particle_id}_contour_{index}_row_col_px'] = loop + [rstart, cstart]
        row = dict(particle_id=particle_id, measurement_status=status,
            touches_image_border=bool(border), complete_particle_candidate=bool(status == 'ok'),
            connectivity=4, bbox_min_row=r0, bbox_min_col=c0,
            bbox_max_row_exclusive=r1, bbox_max_col_exclusive=c1,
            contour_source='chan_vese_zero_level_set', contour_count=len(loops),
            hole_count=holes if holes is not None else '',
            contour_solid_area_px2=net_area if net_area is not None else '',
            contour_envelope_area_px2=envelope_area if envelope_area is not None else '',
            **calibration)
        for prefix, area in (('contour_solid', net_area), ('contour_envelope', envelope_area)):
            row[prefix + '_area_um2'] = area * pixel_area_um2 if area is not None and pixel_area_um2 is not None else ''
            row[prefix + '_equivalent_diameter_um'] = (
                float(np.sqrt(4 * area * pixel_area_um2 / np.pi))
                if area is not None and pixel_area_um2 is not None else '')
        rows.append(row)
    return rows, labels.astype(np.int32), contours


def _save_level_set(mask_file, level_set):
    file = mask_file.with_name(mask_file.stem.removesuffix('_mask') + '_level_set.npy')
    with open(file, 'xb') as f:
        np.save(f, level_set, allow_pickle=False)
    if not np.array_equal(np.load(file, allow_pickle=False), level_set):
        raise RuntimeError('Saved level set failed exact verification.')
    return file


def _measure_record(record_id, record, folder, metadata_rows, distribution_options):
    """Write ONLY Chan-Vese contour measurements and their size distribution."""
    from extracting_discriptors import _distribution_signature, _write_distribution_files
    if record['settings']['method'] != 'CHAN_VESE':
        raise ValueError('Only Chan-Vese has contour measurements in this module.')
    calibration = _pixel_calibration(metadata_rows)
    mask_file = folder / record['mask_file']
    level_file = folder / record['level_set_file'] if record.get('level_set_file') else None
    if record['settings']['method'] == 'CHAN_VESE' and level_file is None:
        raise ValueError('This older Chan-Vese record has no level set; rerun its method/settings.')
    for file, key in ((mask_file, 'mask_sha256'), (level_file, 'level_set_sha256')):
        if file is not None and (not _inside(file.resolve(), folder) or _hash_file(file) != record.get(key)):
            raise ValueError('Segmentation data changed before particle measurement.')
    measurement_id = _hash_json(dict(segmentation_record_id=record_id, calibration=calibration,
        version='2.0', route='contour', connectivity=4, border_policy='flag_and_no_contour_area',
        distribution=_distribution_signature(distribution_options),
        libraries={**_versions(), 'scipy': version('scipy')},
        implementation=inspect.getsource(_particle_regions) + inspect.getsource(_polygon_area_rc)))
    destination = folder / 'contour_measurements'
    destination.mkdir(exist_ok=True)
    prefix = mask_file.stem[:35].replace('.', '_')
    base = destination / (prefix + '__' + measurement_id[:16])
    report = base.with_suffix('.json')
    report_hash = _hash_file(report) if report.exists() else None
    if report.exists():
        audit = json.loads(report.read_text(encoding='utf-8'))
        if audit.get('measurement_id') == measurement_id and all(
                _inside((folder / item['file']).resolve(), folder)
                and (folder / item['file']).is_file() and _hash_file(folder / item['file']) == item['sha256']
                for item in audit.get('outputs', {}).values()) and len(audit.get('outputs', {})) == 5:
            return audit, True
    with Image.open(mask_file) as f:
        mask = np.asarray(f) > 0
    level_set = np.load(level_file, allow_pickle=False) if level_file else None
    rows, labels, contours = _particle_regions(mask, calibration, level_set)
    # Use a new filename if an earlier export was damaged; preserve old files.
    number = 1
    while any(file.exists() for file in (base.with_suffix('.csv'), base.with_suffix('.npz'), base.with_suffix('.tif'), base.with_suffix('.png'), base.with_name(base.name + '_summary.csv'))):
        base = destination / (prefix + '__' + measurement_id[:16] + f'_{number}')
        number += 1
    table, contour_file, label_file = [base.with_suffix(suffix) for suffix in ('.csv', '.npz', '.tif')]
    summary_file, plot = base.with_name(base.name + '_summary.csv'), base.with_suffix('.png')
    context = dict(measurement_id=measurement_id, segmentation_record_id=record_id,
        processed_image=record['source_image'], method_of_segmentation=record['settings']['method'],
        segmentation_parameters=json.dumps(record['settings'], sort_keys=True),
        labelled_image=str(label_file), contour_data=str(contour_file))
    rows = [{**context, **row} for row in rows]
    with open(contour_file, 'xb') as f:
        np.savez_compressed(f, **contours)
    with open(label_file, 'xb') as f:
        Image.fromarray(labels).save(f, format='TIFF', compression='raw')
    with Image.open(label_file) as f:
        if not np.array_equal(np.asarray(f), labels):
            raise RuntimeError('Labelled image failed exact verification.')
    with np.load(contour_file, allow_pickle=False) as saved:
        if set(saved.files) != set(contours) or any(not np.array_equal(saved[k], v) for k, v in contours.items()):
            raise RuntimeError('Contour archive failed exact verification.')
    distribution = _write_distribution_files(table, summary_file, plot, rows, context,
        calibration, distribution_options, 'contour')
    audit = dict(measurement_id=measurement_id, segmentation_record_id=record_id,
        particle_count=len(rows), complete_particle_candidates=sum(row['complete_particle_candidate'] for row in rows),
        calibration=calibration, calibration_source='preprocessing CSV: pixel_size_x_nm/pixel_size_y_nm',
        distribution_summary=distribution,
        objects='4-connected foreground regions; touching particles are not separated',
        area_definitions=dict(solid='foreground area excluding holes', envelope='outer boundary area including holes'),
        outputs={key: dict(file=str(file.relative_to(folder)), sha256=_hash_file(file))
                 for key, file in (('table', table), ('contours', contour_file), ('labels', label_file),
                                   ('summary', summary_file), ('plot', plot))})
    _atomic_json(report, audit, report_hash)
    return audit, False


def _update_particle_csv(folder, audits):
    """Keep the contour route separate from old mixed-measurement exports."""
    from extracting_discriptors import _merge_distribution_audits
    _merge_distribution_audits(folder, folder, audits, 'contour')
    return folder / 'contour_particles.csv'


def _write_segmentation_history(folder, history):
    """Append/upsert an Excel-readable row for each stored image/configuration.

    The JSON remains the detailed algorithm/output cache. This CSV is a separate
    table because the preprocessing CSV/cache supports one row per image.
    Keep existing rows and custom columns, including rows absent from the JSON;
    update managed values by record ID. Identical reruns do not rewrite the file.
    """
    from extracting_discriptors import HISTORY_STATS
    file = folder / 'segmentation_history.csv'
    columns = ['segmentation_record_id', 'processed_image', 'source_path',
        'source_sha256', 'segmented', 'method_of_segmentation',
        'segmentation_parameters', 'threshold_value', 'mask_image',
        'overlay_image', 'level_set_file', 'timestamp_utc', 'error']
    for record in history['records'].values():
        for route in ('contour_psd', 'binary_psd'):
            if isinstance(record.get(route), dict):
                columns += [route + '_' + key for key in HISTORY_STATS if route + '_' + key not in columns]
    before_rows, fields = [], list(columns)
    expected_hash = _hash_file(file) if file.exists() else None
    if file.exists():
        with open(file, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames
            before_rows = list(reader)
        if (not old_fields or len(old_fields) != len(set(old_fields))
                or 'segmentation_record_id' not in old_fields
                or any(None in row or any(v is None for v in row.values()) for row in before_rows)):
            raise ValueError('Malformed segmentation_history.csv; it was not overwritten.')
        fields = old_fields + [name for name in columns if name not in old_fields]
    else:
        old_fields = []
    by_id = {}
    for row in before_rows:
        key = row['segmentation_record_id']
        if not key or key in by_id:
            raise ValueError('Missing/duplicate segmentation record IDs in history CSV.')
        by_id[key] = dict(row)
    for record_id, record in history['records'].items():
        settings = record.get('settings', {})
        row = by_id.setdefault(record_id, {})
        row.update(segmentation_record_id=record_id,
            processed_image=record.get('source_image') or Path(record.get('source_path', '')).name,
            source_path=record.get('source_path', ''),
            source_sha256=record.get('source_sha256', ''),
            segmented='yes' if record.get('status') == 'success' else 'no',
            method_of_segmentation=settings.get('method', 'UNKNOWN'),
            segmentation_parameters=json.dumps(settings, sort_keys=True),
            threshold_value='' if record.get('threshold_value') is None else str(record['threshold_value']),
            mask_image=str(folder / record['mask_file']) if record.get('mask_file') else '',
            overlay_image=str(folder / record['overlay_file']) if record.get('overlay_file') else '',
            level_set_file=str(folder / record['level_set_file']) if record.get('level_set_file') else '',
            timestamp_utc=record.get('timestamp_utc', ''), error=record.get('error', ''))
        for route in ('contour_psd', 'binary_psd'):
            if isinstance(record.get(route), dict):
                row.update({route + '_' + key: str(record[route].get(key, '')) for key in HISTORY_STATS})
    rows = [{key: row.get(key, '') for key in fields} for row in by_id.values()]
    if fields != old_fields or rows != before_rows:
        _write_csv(file, rows, fields, expected_hash)
    return file, len(rows)


def _update_segmentation_summary(rows, image_column, root, history):
    """Keep a cumulative summary; a later method/failure never erases a success."""
    by_source = {}
    for record in history['records'].values():
        if record.get('status') != 'success' or not record.get('source_path'):
            continue
        identity = os.path.normcase(str(Path(record['source_path']).resolve()))
        method = record.get('settings', {}).get('method')
        if method:
            methods = by_source.setdefault(identity, [])
            if method not in methods:
                methods.append(method)
    for row in rows:
        if not row.get(image_column):
            continue
        identity = os.path.normcase(str((root / row[image_column]).resolve()))
        methods = []
        if row.get('segmented') == 'yes':
            for method in row.get('method_of_segmentation', '').split(';'):
                method = method.strip()
                if method and method != 'None' and method not in methods:
                    methods.append(method)
        for method in by_source.get(identity, []):
            if method not in methods:
                methods.append(method)
        row['segmented'] = 'yes' if methods else 'no'
        row['method_of_segmentation'] = '; '.join(methods) if methods else 'None'


def _sync_preprocessing_history(file, rows, image_column):
    """Preserve cumulative segmentation methods when preprocessing runs again.

    The previous stage keeps its CSV projection in a schema-2 JSON record cache.
    Synchronising the same two keys prevents that projection discarding updates.
    Old workflows without this cache require no extra file changes.
    """
    if not file.exists():
        return
    digest = _hash_file(file)
    data = json.loads(file.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('schema') != 2 or not isinstance(data.get('records'), list):
        raise ValueError('Unrecognised preprocessing history; refuse to change CSV until it can be preserved.')
    by_output = {row[image_column]: row for row in rows if row.get(image_column)}
    changed = False
    for record in data['records']:
        row = by_output.get(record.get('preprocess.output_image'))
        if row is None:
            continue
        for column in SEGMENTATION_COLUMNS:
            if record.get(column) != row[column]:
                record[column] = row[column]
                changed = True
    if changed:
        _atomic_json(file, data, digest)


def segment_images(images, output_folder=None, *, method='OTSU', foreground='bright',
                   nbins=256, overlay_alpha=0.35, recursive=False,
                   csv_name='preprocessing_metadata.csv', chan_vese_parameters=None,
                   measure_particles=True, percentile_range=(10, 90),
                   area_basis='solid', exclude_border=True, histogram_bins=30):
    """Segment all previous-stage image records and update the SAME CSV.

    images is the preprocessing output folder (not its segmented subfolder).
    A TIFF mask and RGB overlay are saved under images/segmented by default.
    Select method='OTSU' (default) or method='CHAN_VESE' (also 'chan-vese').
    The input CSV keeps two summary columns: segmented=yes/no and a cumulative
    method_of_segmentation such as 'OTSU; CHAN_VESE'. These describe successful
    segmentations performed, not just the latest request. A failed method never
    clears a previous success. For the status/parameters of each configuration,
    use output_folder/segmentation_history.csv (one row per record ID).
    Existing JSON records are exported automatically, including older methods.
    Identical reruns update/reuse a row; different settings create another row.
    Full numerical diagnostics remain in segmentation_records.json.
    measure_particles=True measures ONLY retained Chan-Vese contours. It has no
    effect for Otsu. Binary-mask areas for either method belong to the separate
    extracting_discriptors.particle_size_distribution function.
    Contour results: contour_particles.csv, contour_psd_summary.csv and per-record
    labelled TIFFs, retained contour NPZs, tables and plots in contour_measurements/.
    Distribution options: percentile_range=(10, 90), area_basis='solid' (or
    'envelope'), exclude_border=True, histogram_bins=30. Percentiles describe
    number-weighted equivalent circular diameters; they do not trim the sample.
    segmentation_history.csv receives contour_psd_* fields; binary_psd_* fields
    written by extracting_discriptors are retained. Full distributions keep all
    options/calibration variants. No pooling across images or segmentation recipes.
    Pixel sizes come from CSV pixel_size_x_nm/pixel_size_y_nm for each image.
    Missing/invalid calibration leaves physical values blank; pixel areas remain.
    Objects use 4-connectivity; touching particles are not automatically split.
    Border-cut regions are flagged and have no contour area. Solid and envelope
    areas are both exported so holes are explicitly accounted for.
    foreground='bright'/'dark' selects the brighter/darker resulting region.

    For Chan-Vese, supply chan_vese_parameters=None for defaults or a dictionary
    overriding mu, lambda1, lambda2, tol, max_num_iter, dt, init_level_set.
    Example: method='CHAN_VESE', chan_vese_parameters={'mu': 0.1}.
    nbins applies only to Otsu. Chan-Vese has no intensity threshold; its recorded
    threshold_value and nbins are null. Otsu rejects Chan-Vese options to avoid
    silently ignoring a requested setting. Both methods use scikit-image.
    All existing CSV cells remain unchanged except these two owned columns.

    Repeated identical requests skip verified outputs. Changed input/settings
    produce separate files while retaining earlier segmentation versions.
    recursive=False selects CSV-referenced TIFFs directly in images; True also
    selects CSV-referenced images in its subfolders. It never scans masks/overlays.
    """
    root = Path(images).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('images must be the preprocessing output folder.')
    folder = (root / 'segmented') if output_folder is None else Path(output_folder).expanduser().resolve()
    folder = folder.resolve()
    if folder == root or not _inside(folder, root):
        raise ValueError('output_folder must be a new subfolder of the input folder.')
    method = str(method).strip().upper().replace('-', '_').replace(' ', '_')
    if method not in ('OTSU', 'CHAN_VESE'):
        raise ValueError("method must be 'OTSU' or 'CHAN_VESE'.")
    if method == 'OTSU' and chan_vese_parameters is not None:
        raise ValueError("chan_vese_parameters applies only to method='CHAN_VESE'.")
    cv_settings = _chan_vese_parameters(chan_vese_parameters) if method == 'CHAN_VESE' else None
    from extracting_discriptors import _options
    distribution_options = _options(percentile_range, area_basis, exclude_border, histogram_bins)
    if not isinstance(measure_particles, bool):
        raise ValueError('measure_particles must be True or False.')
    if foreground not in ('bright', 'dark'):
        raise ValueError("foreground must be 'bright' or 'dark'.")
    if isinstance(nbins, (bool, np.bool_)) or not isinstance(nbins, (int, np.integer)) or nbins < 2:
        raise ValueError('nbins must be an integer of at least 2.')
    if (not isinstance(overlay_alpha, (float, int)) or isinstance(overlay_alpha, bool)
            or not np.isfinite(overlay_alpha) or not 0 <= overlay_alpha <= 1):
        raise ValueError('overlay_alpha must be between 0 and 1.')
    if Path(csv_name).name != csv_name or Path(csv_name).suffix.lower() != '.csv':
        raise ValueError('csv_name must be a CSV filename.')
    csv_file = root / csv_name
    # Share the previous stage's lock because both modify this folder's CSV/cache.
    lock = root / '.preprocessing.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError('Preprocessing/segmentation is running, or a stale .preprocessing.lock exists.') from exc
    with os.fdopen(fd, 'w') as f:
        f.write(str(os.getpid()))
    try:
        fields, rows, image_column, csv_hash = _read_csv(csv_file)
        before_rows = [dict(row) for row in rows]
        before_fields = list(fields)
        for column in SEGMENTATION_COLUMNS:
            if column not in fields:
                fields.append(column)
            for row in rows:
                if not row.get(column):
                    row[column] = 'no' if column == 'segmented' else 'None'
        grouped = {}
        for i, row in enumerate(rows):
            name = row.get(image_column, '')
            if not name:
                continue
            file = (root / name).resolve()
            if not _inside(file, root):
                raise ValueError('CSV points outside the preprocessing output folder.')
            if _inside(file, folder):
                continue
            if not recursive and file.parent != root:
                continue
            grouped.setdefault(file, []).append(i)
        if not grouped:
            raise ValueError('No CSV-referenced preprocessing images match the selected folder/recursive setting.')
        folder.mkdir(parents=True, exist_ok=True)
        history_file = folder / 'segmentation_records.json'
        history_hash = _hash_file(history_file) if history_file.exists() else None
        if history_file.exists():
            history = json.loads(history_file.read_text(encoding='utf-8'))
            if not isinstance(history, dict) or history.get('schema') != 1:
                raise ValueError('Unrecognised segmentation history; nothing was overwritten.')
        else:
            history = dict(schema=1, records={}, active_outputs={})
        if not isinstance(history.get('records'), dict) or not isinstance(history.get('active_outputs'), dict):
            raise ValueError('Malformed segmentation history.')
        libraries = _versions()
        settings = dict(method=method, foreground=foreground,
            nbins=int(nbins) if method == 'OTSU' else None,
            overlay_alpha=float(overlay_alpha))
        implementation = inspect.getsource(otsu_thresholding)
        if method == 'CHAN_VESE':
            settings['chan_vese_parameters'] = cv_settings
            implementation = (inspect.getsource(chan_vese_segmentation)
                              + inspect.getsource(_chan_vese_parameters))
        recipe_hash = _hash_json(dict(version=VERSION, settings=settings, libraries=libraries,
            implementation=implementation + inspect.getsource(_make_overlay)))
        summary = dict(input_count=len(grouped), segmented=0, skipped=0, failed=0,
                       output_folder=str(folder), csv_path=str(csv_file), errors=[],
                       measurements_written=0, measurements_skipped=0,
                       measurement_failed=0, measurement_errors=[], calibration_warnings=[])
        history_changed = False
        measurement_audits = []
        for source, indices in sorted(grouped.items(), key=lambda pair: str(pair[0])):
            source_hash = ''
            identity = os.path.normcase(str(source))
            record_id = _hash_json([identity, 'unreadable', recipe_hash])
            try:
                if source.suffix.lower() not in ('.tif', '.tiff') or source.name.lower().endswith('.analysis_mask.tif'):
                    raise ValueError('Expected a specimen TIFF, not an analysis mask.')
                source_hash = _hash_file(source)
                record_id = _hash_json([identity, source_hash, recipe_hash])
                previous = history['records'].get(record_id)
                if _valid_outputs(previous, folder, source_hash, recipe_hash):
                    summary['skipped'] += 1
                else:
                    image, resolution = _load_image(source)
                    diagnostics, level_set = None, None
                    if method == 'OTSU':
                        mask, threshold_value = otsu_thresholding(image, foreground=foreground, nbins=nbins)
                    else:
                        mask, diagnostics, level_set = chan_vese_segmentation(image, foreground=foreground,
                            chan_vese_parameters=cv_settings, return_details=True, return_level_set=True)
                        threshold_value = None
                    metadata_row = rows[indices[0]]
                    calibration = {k: metadata_row[k] for k in ('pixel_size_x_nm', 'pixel_size_y_nm', 'calibration_status') if metadata_row.get(k)}
                    mask_file, overlay_file = _save_outputs(image, mask, threshold_value, source, folder,
                        record_id, method, foreground, settings['nbins'], overlay_alpha,
                        resolution, calibration, settings=settings)
                    level_file = _save_level_set(mask_file, level_set) if level_set is not None else None
                    if _hash_file(source) != source_hash:
                        raise RuntimeError('Input image changed during segmentation; rerun after preprocessing completes.')
                    history['records'][record_id] = dict(status='success', source_image=source.name,
                        source_path=str(source), source_sha256=source_hash, recipe_sha256=recipe_hash,
                        settings=settings, threshold_value=threshold_value,
                        diagnostics=diagnostics, image_shape=list(image.shape),
                        input_dtype=str(image.dtype), foreground_pixels=int(mask.sum()),
                        foreground_fraction=float(mask.mean()), mask_dtype='uint8',
                        mask_values=dict(background=0, foreground=255), calibration=calibration,
                        mask_file=mask_file.name, overlay_file=overlay_file.name,
                        mask_sha256=_hash_file(mask_file), overlay_sha256=_hash_file(overlay_file),
                        timestamp_utc=datetime.now(timezone.utc).isoformat(), version=VERSION,
                        libraries=libraries, verification=dict(mask_pixels_match=True, overlay_pixels_match=True, input_unchanged=True))
                    if level_file is not None:
                        history['records'][record_id].update(level_set_file=level_file.name,
                            level_set_sha256=_hash_file(level_file), level_set_positive_region='selected_foreground')
                    history_changed = True
                    summary['segmented'] += 1
                if history['active_outputs'].get(identity) != record_id:
                    history['active_outputs'][identity] = record_id
                    history_changed = True
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                # Preserve the earlier success if repairing its outputs fails.
                # The failed repair gets a separate audit entry, not a deletion.
                if history['records'].get(record_id, {}).get('status') == 'success':
                    record_id = _hash_json([record_id, 'failed_repair',
                                           datetime.now(timezone.utc).isoformat()])
                history['records'][record_id] = dict(status='failed', source_path=str(source),
                    source_sha256=source_hash, recipe_sha256=recipe_hash, settings=settings,
                    error=error, timestamp_utc=datetime.now(timezone.utc).isoformat())
                history['active_outputs'].pop(identity, None)
                history_changed = True
                summary['failed'] += 1
                summary['errors'].append(dict(input_file=str(source), error=error))
                continue
            if measure_particles and method == 'CHAN_VESE':
                try:
                    audit, cached = _measure_record(record_id, history['records'][record_id], folder,
                                                    [rows[i] for i in indices], distribution_options)
                    measurement_audits.append(audit)
                    if history['records'][record_id].get('contour_psd') != audit['distribution_summary']:
                        history['records'][record_id]['contour_psd'] = audit['distribution_summary']
                        history_changed = True
                    summary['measurements_skipped' if cached else 'measurements_written'] += 1
                    if not audit['calibration']['calibration_valid']:
                        summary['calibration_warnings'].append(dict(input_file=str(source),
                            warning=audit['calibration']['calibration_note']))
                except Exception as exc:
                    # A measurement problem must not erase successful segmentation.
                    summary['measurement_failed'] += 1
                    summary['measurement_errors'].append(dict(input_file=str(source),
                        error=f'{type(exc).__name__}: {exc}'))
        if _hash_file(csv_file) != csv_hash:
            raise RuntimeError('Preprocessing CSV changed during segmentation/contour measurement; rerun with the latest calibration.')
        if history_changed or not history_file.exists():
            _atomic_json(history_file, history, history_hash)
        history_csv, history_count = _write_segmentation_history(folder, history)
        summary['history_csv_path'] = str(history_csv)
        summary['history_record_count'] = history_count
        if measure_particles and method == 'CHAN_VESE':
            particle_csv = _update_particle_csv(folder, measurement_audits)
            summary['particle_csv_path'] = str(particle_csv) if particle_csv.exists() else None
            summary['psd_csv_path'] = str(folder / 'contour_psd_summary.csv')
        _update_segmentation_summary(rows, image_column, root, history)
        # Synchronise only these two columns in the previous stage's record cache,
        # so a later preprocess_images call retains the segmentation flags.
        _sync_preprocessing_history(csv_file.with_suffix('.records.json'), rows, image_column)
        if fields != before_fields or rows != before_rows:
            _write_csv(csv_file, rows, fields, csv_hash)
        return summary
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', help='Previous preprocessing output folder')
    parser.add_argument('--output', help='Segmentation subfolder; default: images/segmented')
    parser.add_argument('--method', default='OTSU', help='OTSU or CHAN_VESE')
    parser.add_argument('--chan-vese-parameters', type=json.loads,
        help='JSON dictionary of Chan-Vese overrides, e.g. {"mu":0.1,"max_num_iter":500}')
    parser.add_argument('--foreground', choices=('bright', 'dark'), default='bright')
    parser.add_argument('--nbins', type=int, default=256)
    parser.add_argument('--overlay-alpha', type=float, default=0.35)
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--csv-name', default='preprocessing_metadata.csv')
    parser.add_argument('--no-particle-measurements', action='store_true',
        help='Skip Chan-Vese contour size distributions (Otsu never measures here)')
    parser.add_argument('--percentiles', type=float, nargs=2, default=(10, 90))
    parser.add_argument('--area-basis', choices=('solid', 'envelope'), default='solid')
    parser.add_argument('--include-border', action='store_true')
    parser.add_argument('--histogram-bins', type=int, default=30)
    args = parser.parse_args()
    result = segment_images(args.images, args.output, method=args.method, foreground=args.foreground,
        nbins=args.nbins, overlay_alpha=args.overlay_alpha, recursive=args.recursive,
        csv_name=args.csv_name, chan_vese_parameters=args.chan_vese_parameters,
        measure_particles=not args.no_particle_measurements, percentile_range=args.percentiles,
        area_basis=args.area_basis, exclude_border=not args.include_border,
        histogram_bins=args.histogram_bins)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if result['failed'] or result['measurement_failed'] else 0)
