"""Particle-size distributions from saved binary segmentation masks.

Keep this file beside segmentation.py (the requested filename is intentional).
The binary route counts pixels only: it never extracts contours or level sets.
The shared distribution helpers also serve Chan-Vese's separate contour route.

Example:
    from extracting_discriptors import particle_size_distribution
    summary = particle_size_distribution(images=preprocessed_folder / 'segmented')

One distribution is produced per segmentation record and parameter set. Sizes
are number-weighted 2D equivalent circular diameters, not volume-weighted laser
diffraction D-values or true 3D diameters. Each 4-connected foreground region is
an object; touching particles are NOT separated. No arbitrary size filtering is
applied. Border-cut objects are retained in the table and excluded from the PSD
by default. All exclusions are recorded. Percentiles describe the distribution;
they do not clip or discard particles. Defaults: P10, P90, solid area, 30 bins.

Calibration is read separately for each image from the preprocessing CSV's
pixel_size_x_nm/pixel_size_y_nm. Missing/invalid calibration gives an explicitly
labelled distribution in equivalent pixel units and blank physical areas.
Statistics use micrometres when calibrated. No pooling across images/recipes.

Outputs: binary_particles.csv, binary_psd_summary.csv, per-record labelled TIFFs,
particle/summary tables and histogram + empirical cumulative distribution PNGs.
segmentation_history.csv is extended with binary_psd_* summary columns; contour
columns are retained. Full tables keep all parameter/calibration variants.
Close CSV files in Excel before running either processing stage.
"""
from pathlib import Path
from importlib.metadata import version
import argparse
import csv
import inspect
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage

PSD_VERSION = '1.0.0'
HISTORY_STATS = ('distribution_id', 'measurement_route', 'area_basis', 'size_unit',
    'particle_count_total', 'particle_count_used', 'border_particles_excluded',
    'invalid_particles_excluded', 'mean_particle_size', 'median_particle_size',
    'std_particle_size', 'min_particle_size', 'max_particle_size',
    'lower_percentile', 'lower_percentile_size', 'upper_percentile',
    'upper_percentile_size', 'p10', 'p50', 'p90', 'distribution_plot', 'summary_csv')


def _options(percentile_range=(10, 90), area_basis='solid', exclude_border=True,
             histogram_bins=30):
    try:
        bounds = np.asarray(percentile_range, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError('percentile_range must contain two numbers.') from exc
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or not 0 <= bounds[0] < bounds[1] <= 100):
        raise ValueError('Percentiles must satisfy 0 <= lower < upper <= 100.')
    if area_basis not in ('solid', 'envelope'):
        raise ValueError("area_basis must be 'solid' or 'envelope'.")
    if not isinstance(exclude_border, bool):
        raise ValueError('exclude_border must be True or False.')
    if isinstance(histogram_bins, bool) or not isinstance(histogram_bins, int) or histogram_bins < 1:
        raise ValueError('histogram_bins must be a positive integer.')
    return dict(percentile_range=bounds.tolist(), area_basis=area_basis,
                exclude_border=exclude_border, histogram_bins=histogram_bins)


def _distribution_signature(options):
    import segmentation as seg
    return dict(version=PSD_VERSION, options=options,
        libraries={p: version(p) for p in ('numpy', 'scipy', 'matplotlib', 'Pillow')},
        implementation=seg._hash_json(inspect.getsource(summarize_particle_sizes)
            + inspect.getsource(_write_distribution_files) + inspect.getsource(_plot_distribution)))


def summarize_particle_sizes(sizes, percentile_range=(10, 90)):
    """Number-weighted diameter statistics; empty/single-object cases explicit.

    Percentiles use NumPy's linear interpolation. Standard deviation is sample
    SD (ddof=1), left blank for fewer than two objects. No percentile trimming.
    """
    bounds = _options(percentile_range)['percentile_range']
    values = np.asarray(sizes, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('sizes must be a 1D array of finite positive diameters.')
    result = dict(particle_count_used=int(values.size),
        lower_percentile=bounds[0], upper_percentile=bounds[1],
        mean_particle_size='', median_particle_size='', std_particle_size='',
        min_particle_size='', max_particle_size='', lower_percentile_size='',
        upper_percentile_size='', p10='', p50='', p90='')
    if values.size:
        low, high, p10, p50, p90 = np.percentile(values, [*bounds, 10, 50, 90], method='linear')
        result.update(mean_particle_size=float(values.mean()),
            median_particle_size=float(np.median(values)),
            std_particle_size=float(values.std(ddof=1)) if values.size > 1 else '',
            min_particle_size=float(values.min()), max_particle_size=float(values.max()),
            lower_percentile_size=float(low), upper_percentile_size=float(high),
            p10=float(p10), p50=float(p50), p90=float(p90))
    return result


def _plot_distribution(file, sizes, summary, bins):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(10, 4), constrained_layout=True)
    FigureCanvasAgg(figure)
    hist, cumulative = figure.subplots(1, 2)
    label = 'Equivalent circular diameter (' + summary['size_unit'] + ')'
    if len(sizes):
        hist.hist(sizes, bins=bins, color='#386a9f', edgecolor='white', linewidth=0.4)
        ordered = np.sort(sizes)
        # Empirical cumulative counts; summary percentiles use linear interpolation.
        cumulative.step(np.r_[ordered[0], ordered], np.r_[0, np.arange(1, len(ordered) + 1) / len(ordered) * 100], where='post', color='#386a9f')
        marks = [(summary['lower_percentile'], summary['lower_percentile_size'], '#d28b28'),
                 (50, summary['p50'], '#328267'),
                 (summary['upper_percentile'], summary['upper_percentile_size'], '#b64c55')]
        for percentile, value, color in marks:
            hist.axvline(value, color=color, linestyle='--', label=f'P{percentile:g}: {value:.3g}')
            cumulative.axvline(value, color=color, linestyle='--', linewidth=1)
        hist.legend(fontsize=8)
    else:
        for ax in (hist, cumulative):
            ax.text(0.5, 0.5, 'No objects eligible for this distribution',
                    ha='center', va='center', transform=ax.transAxes, wrap=True)
    hist.set(xlabel=label, ylabel='Object count', title='Number-weighted distribution')
    cumulative.set(xlabel=label, ylabel='Cumulative object count (%)', ylim=(0, 102), title='Empirical cumulative distribution')
    title = f"{summary['processed_image'][:55]} | {summary['method_of_segmentation']} | {summary['measurement_route']}"
    figure.suptitle(title, fontsize=10)
    with open(file, 'xb') as f:
        figure.savefig(f, format='png', dpi=180, metadata={'Software': 'SEM descriptors'})
    figure.clear()


def _write_distribution_files(table, summary_file, plot, rows, context, calibration,
                              options, route):
    """Shared statistics/plotting only; areas are supplied by separate routes."""
    import segmentation as seg
    sizes, output, border_excluded, invalid_excluded = [], [], 0, 0
    unit = 'um' if calibration['calibration_valid'] else 'px'
    prefix = ('contour_' if route == 'contour' else 'mask_') + options['area_basis']
    for original in rows:
        row = dict(original)
        area_px = row.get(prefix + '_area_px2', '')
        valid = isinstance(area_px, (int, float, np.integer, np.floating)) and np.isfinite(area_px) and area_px > 0
        border = bool(row['touches_image_border'])
        included = valid and not (border and options['exclude_border'])
        reason = '' if included else ('border_truncated' if border and options['exclude_border'] else 'invalid_or_open_contour')
        if not included:
            if border and options['exclude_border']:
                border_excluded += 1
            else:
                invalid_excluded += 1
        size = ''
        if valid:
            area = row[prefix + '_area_um2'] if unit == 'um' else area_px
            size = float(np.sqrt(4 * area / np.pi))
        row.update(measurement_route=route, area_basis=options['area_basis'],
            equivalent_circular_diameter=size, size_unit=unit,
            included_in_distribution=included, exclusion_reason=reason)
        output.append(row)
        if included:
            sizes.append(size)
    summary = {**context, **calibration, **summarize_particle_sizes(sizes, options['percentile_range']),
        'distribution_id': context['measurement_id'], 'measurement_route': route,
        'area_basis': options['area_basis'], 'size_unit': unit,
        'distribution_basis': 'number_weighted_2d_equivalent_circular_diameter',
        'particle_count_total': len(rows), 'border_particles_excluded': border_excluded,
        'invalid_particles_excluded': invalid_excluded,
        'exclude_border': options['exclude_border'], 'histogram_bins': options['histogram_bins'],
        'percentile_method': 'linear', 'standard_deviation_ddof': 1,
        'particle_table': str(table), 'summary_csv': str(summary_file), 'distribution_plot': str(plot)}
    columns = list(output[0]) if output else list(context) + ['particle_id', 'included_in_distribution']
    seg._write_csv(table, output, columns, None)
    seg._write_csv(summary_file, [summary], list(summary), None)
    _plot_distribution(plot, sizes, summary, options['histogram_bins'])
    return summary


def _merge_csv(file, additions, keys):
    """Upsert by stable IDs; retain other configurations and user-added columns."""
    import segmentation as seg
    before, fields = [], []
    digest = seg._hash_file(file) if file.exists() else None
    if file.exists():
        with open(file, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            fields, before = reader.fieldnames, list(reader)
        if (not fields or len(fields) != len(set(fields)) or not set(keys) <= set(fields)
                or any(None in row or any(v is None for v in row.values()) for row in before)):
            raise ValueError(f'Malformed {file.name}; it was not overwritten.')
    old_fields = list(fields)
    by_id = {}
    for row in before:
        key = tuple(row[k] for k in keys)
        if key in by_id:
            raise ValueError(f'Duplicate IDs in {file.name}.')
        by_id[key] = dict(row)
    for row in additions:
        values = {k: '' if v is None else str(v) for k, v in row.items()}
        fields += [k for k in values if k not in fields]
        key = tuple(values[k] for k in keys)
        by_id.setdefault(key, {}).update(values)
    result = [{k: row.get(k, '') for k in fields} for row in by_id.values()]
    if fields and (not file.exists() or fields != old_fields or result != before):
        seg._write_csv(file, result, fields, digest)


def _merge_distribution_audits(segmentation_folder, destination, audits, route):
    rows = []
    for audit in audits:
        with open(segmentation_folder / audit['outputs']['table']['file'], newline='', encoding='utf-8-sig') as f:
            rows.extend(csv.DictReader(f))
    _merge_csv(destination / f'{route}_particles.csv', rows, ('measurement_id', 'particle_id'))
    _merge_csv(destination / f'{route}_psd_summary.csv', [a['distribution_summary'] for a in audits], ('distribution_id',))


def _binary_regions(mask, calibration):
    """Pixel counting only; no contour extraction in the binary route."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError('Expected a non-empty 2D binary mask.')
    labels, _ = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, 1))
    rows = []
    factor = calibration['pixel_size_x_nm'] * calibration['pixel_size_y_nm'] / 1e6 if calibration['calibration_valid'] else None
    if factor is not None and (not np.isfinite(factor) or factor <= 0):
        raise ValueError('Unusable physical pixel area.')
    for particle_id, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        component = labels[box] == particle_id
        solid = int(component.sum())
        envelope = int(ndimage.binary_fill_holes(component, structure=np.ones((3, 3), dtype=bool)).sum())
        r0, r1, c0, c1 = box[0].start, box[0].stop, box[1].start, box[1].stop
        border = r0 == 0 or c0 == 0 or r1 == mask.shape[0] or c1 == mask.shape[1]
        rows.append(dict(particle_id=particle_id, touches_image_border=border,
            connectivity=4, bbox_min_row=r0, bbox_min_col=c0, bbox_max_row_exclusive=r1,
            bbox_max_col_exclusive=c1, mask_solid_area_px2=solid, mask_envelope_area_px2=envelope,
            mask_solid_area_um2=solid * factor if factor is not None else '',
            mask_envelope_area_um2=envelope * factor if factor is not None else '', **calibration))
    return rows, labels.astype(np.int32)


def _measure_binary_record(record_id, record, folder, destination, metadata_rows, options):
    import segmentation as seg
    calibration = seg._pixel_calibration(metadata_rows)
    file = (folder / record['mask_file']).resolve()
    if not seg._inside(file, folder) or seg._hash_file(file) != record.get('mask_sha256'):
        raise ValueError('Binary mask is missing, outside the folder, or has changed; rerun segmentation.')
    measurement_id = seg._hash_json(dict(segmentation_record_id=record_id,
        route='binary', mask_sha256=record['mask_sha256'], calibration=calibration,
        distribution=_distribution_signature(options), implementation=inspect.getsource(_binary_regions)))
    prefix = record_id[:12] + '__binary__' + measurement_id[:16]
    base = destination / prefix
    organise_json_records(destination)
    report = json_record_path(base.with_suffix('.json'))
    report_hash = seg._hash_file(report) if report.exists() else None
    if report.exists():
        audit = json.loads(report.read_text(encoding='utf-8'))
        if audit.get('measurement_id') == measurement_id and len(audit.get('outputs', {})) == 4 and all(
                seg._inside((folder / item['file']).resolve(), folder)
                and (folder / item['file']).is_file() and seg._hash_file(folder / item['file']) == item['sha256']
                for item in audit['outputs'].values()):
            return audit, True
    with Image.open(file) as image:
        pixels = np.asarray(image)
    if pixels.ndim != 2 or not set(np.unique(pixels)) <= {0, 255}:
        raise ValueError('Expected the saved 0/255 binary segmentation mask.')
    rows, labels = _binary_regions(pixels > 0, calibration)
    number = 1
    while any(base.with_suffix(suffix).exists() for suffix in ('.csv', '.tif', '.png')) or base.with_name(base.name + '_summary.csv').exists():
        base = destination / (prefix + f'_{number}')
        number += 1
    table, label_file, plot = [base.with_suffix(s) for s in ('.csv', '.tif', '.png')]
    summary_file = base.with_name(base.name + '_summary.csv')
    with open(label_file, 'xb') as f:
        Image.fromarray(labels).save(f, format='TIFF', compression='raw')
    with Image.open(label_file) as saved:
        if not np.array_equal(np.asarray(saved), labels):
            raise RuntimeError('Labelled binary image failed verification.')
    context = dict(measurement_id=measurement_id, segmentation_record_id=record_id,
        processed_image=record.get('source_image') or Path(record['source_path']).name,
        method_of_segmentation=record['settings']['method'],
        segmentation_parameters=json.dumps(record['settings'], sort_keys=True),
        source_mask=str(file), labelled_image=str(label_file))
    rows = [{**context, **r} for r in rows]
    summary = _write_distribution_files(table, summary_file, plot, rows, context, calibration, options, 'binary')
    audit = dict(measurement_id=measurement_id, segmentation_record_id=record_id,
        calibration=calibration, distribution_summary=summary,
        outputs={k: dict(file=str(f.relative_to(folder)), sha256=seg._hash_file(f))
                 for k, f in (('table', table), ('labels', label_file), ('plot', plot), ('summary', summary_file))})
    seg._atomic_json(report, audit, report_hash)
    return audit, False


def particle_size_distribution(images, output_folder=None, *, input_metadata_csv=None,
                               percentile_range=(10, 90), area_basis='solid',
                               exclude_border=True, histogram_bins=30, methods=None,
                               show_progress=True):
    """Measure all saved successful binary-mask configurations, not overlays.

    images is the segmented folder (or its preprocessing parent). Metadata CSV
    defaults to preprocessing_metadata.csv in a parent directory; set
    input_metadata_csv explicitly for another name/location. Each mask is matched
    by the original preprocessing image path, never by a shared/global scale.
    methods=None includes OTSU and CHAN_VESE; a string or list restricts methods.
    No images, contours or segmentation settings are altered by this function.
    """
    import segmentation as seg
    options = _options(percentile_range, area_basis, exclude_border, histogram_bins)
    folder = Path(images).expanduser().resolve()
    if not ((folder / 'segmentation_records.json').is_file()
            or (folder / 'json' / 'segmentation_records.json').is_file()):
        folder = folder / 'segmented'
    if not ((folder / 'segmentation_records.json').is_file()
            or (folder / 'json' / 'segmentation_records.json').is_file()):
        raise FileNotFoundError('Point images at the segmented folder or its preprocessing parent.')
    if input_metadata_csv is None:
        metadata_csv = next((p / 'preprocessing_metadata.csv' for p in folder.parents
                             if (p / 'preprocessing_metadata.csv').is_file()), None)
        if metadata_csv is None:
            raise FileNotFoundError('Set input_metadata_csv to the preprocessing CSV.')
    else:
        metadata_csv = Path(input_metadata_csv).expanduser().resolve()
    root = metadata_csv.parent.resolve()
    destination = folder / 'binary_descriptors' if output_folder is None else Path(output_folder).expanduser().resolve()
    if destination == folder or not seg._inside(destination, folder):
        raise ValueError('output_folder must be a subfolder of the segmentation folder.')
    selected = None
    if methods is not None:
        methods = [methods] if isinstance(methods, str) else list(methods)
        selected = {str(m).strip().upper().replace('-', '_').replace(' ', '_') for m in methods}
        if not selected or not selected <= {'OTSU', 'CHAN_VESE'}:
            raise ValueError('methods must select OTSU and/or CHAN_VESE.')
    lock = root / '.preprocessing.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError('Another preprocessing/segmentation/descriptor run is active, or its lock is stale.') from exc
    with os.fdopen(fd, 'w') as f:
        f.write(str(os.getpid()))
    progress = None
    try:
        organise_json_records(folder)
        organise_json_records(destination)
        history_file = json_record_path(folder / 'segmentation_records.json')
        _, metadata, image_column, metadata_hash = seg._read_csv(metadata_csv)
        by_image = {}
        for row in metadata:
            if row.get(image_column):
                key = os.path.normcase(str((root / row[image_column]).resolve()))
                by_image.setdefault(key, []).append(row)
        history_hash = seg._hash_file(history_file)
        history = json.loads(history_file.read_text(encoding='utf-8'))
        if not isinstance(history, dict) or history.get('schema') != 1 or not isinstance(history.get('records'), dict):
            raise ValueError('Unrecognised segmentation history.')
        destination.mkdir(parents=True, exist_ok=True)
        audits, changed = [], False
        result = dict(input_count=0, processed=0, skipped=0, failed=0, errors=[], calibration_warnings=[],
            output_folder=str(destination), particle_csv_path=str(destination / 'binary_particles.csv'),
            psd_csv_path=str(destination / 'binary_psd_summary.csv'), history_csv_path=str(folder / 'segmentation_history.csv'))
        selected_records = [(key, rec) for key, rec in history['records'].items()
                            if rec.get('status') == 'success'
                            and (selected is None or rec.get('settings', {}).get('method') in selected)]
        progress = Progress(len(selected_records), 'Binary descriptors', enabled=show_progress, unit='records').start()
        for record_id, record in progress.items(selected_records, name=lambda pair: pair[1].get('source_image', pair[0][:12])):
            result['input_count'] += 1
            try:
                key = os.path.normcase(str(Path(record['source_path']).resolve()))
                rows = by_image.get(key)
                if rows is None:
                    raise ValueError('No matching preprocessing CSV row for this segmentation; no scale guessed.')
                audit, cached = _measure_binary_record(record_id, record, folder, destination, rows, options)
                audits.append(audit)
                result['skipped' if cached else 'processed'] += 1
                if record.get('binary_psd') != audit['distribution_summary']:
                    record['binary_psd'] = audit['distribution_summary']
                    changed = True
                if not audit['calibration']['calibration_valid']:
                    result['calibration_warnings'].append(dict(segmentation_record_id=record_id,
                        warning=audit['calibration']['calibration_note']))
            except Exception as exc:
                result['failed'] += 1
                result['errors'].append(dict(segmentation_record_id=record_id, error=f'{type(exc).__name__}: {exc}'))
        progress.status('saving descriptor and PSD CSVs')
        if seg._hash_file(metadata_csv) != metadata_hash:
            raise RuntimeError('Preprocessing CSV changed during measurement; rerun with the latest calibration.')
        _merge_distribution_audits(folder, destination, audits, 'binary')
        if changed:
            seg._atomic_json(history_file, history, history_hash)
        seg._write_segmentation_history(folder, history)
        return result
    finally:
        if progress is not None:
            progress.close()
        lock.unlink(missing_ok=True)


from sem_paths import json_record_path, organise_json_records
from sem_progress import Progress, instrument_functions
instrument_functions(globals(), exclude=('particle_size_distribution',),
    include_private=('_binary_regions', '_measure_binary_record', '_write_distribution_files', '_plot_distribution'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', help='Segmentation output folder or its preprocessing parent')
    parser.add_argument('--output')
    parser.add_argument('--input-metadata-csv')
    parser.add_argument('--percentiles', type=float, nargs=2, default=(10, 90))
    parser.add_argument('--area-basis', choices=('solid', 'envelope'), default='solid')
    parser.add_argument('--include-border', action='store_true')
    parser.add_argument('--histogram-bins', type=int, default=30)
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--no-progress', action='store_true', help='Disable console progress')
    args = parser.parse_args()
    result = particle_size_distribution(args.images, args.output,
        input_metadata_csv=args.input_metadata_csv, percentile_range=args.percentiles,
        area_basis=args.area_basis, exclude_border=not args.include_border,
        histogram_bins=args.histogram_bins, methods=args.methods, show_progress=not args.no_progress)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if result['failed'] else 0)
