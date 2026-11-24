"""Incremental SEM footer removal and readable metadata CSV export.

Requires the other sem_*.py modules in the same directory. Run on original
TIFFs before normalisation. All files are saved under the selected output folder.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile

from PIL import Image
import numpy as np
from collections import Counter
import sem_metadata
from sem_processing import process_source, load_valid_mask
import sem_information_bar as core

VERSION = '2.0.0'
BASE_COLUMNS = [
    'source_id', 'source_file', 'source_path', 'source_sha256', 'status', 'error',
    'output_image', 'audit_json', 'output_sha256', 'processed_utc',
    'batch_version', 'crop_version', 'original_width_px', 'original_height_px',
    'cropped_width_px', 'cropped_height_px', 'removed_rows', 'dtype',
    'pixel_size_x_nm', 'pixel_size_y_nm', 'accelerating_voltage_kV',
    'working_distance_mm', 'dwell_time_us', 'predicted_beam_current_pA',
    'magnification', 'operator', 'acquisition_date', 'acquisition_time',
    'manufacturer', 'model', 'detector', 'metadata_reader', 'metadata_status',
    'file_format', 'page_count',
    'calibration_status', 'metadata_warnings', 'crop_method', 'review_required',
    'review_reason', 'proposed_crop_row', 'policy_hash', 'review_image',
    'review_image_sha256', 'review_preview', 'analysis_mask', 'analysis_mask_sha256',
    'analysis_included_pixels',
]


def _source_id(path):
    canonical = os.path.normcase(str(Path(path).resolve()))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _input_files(images, output_folder, recursive):
    entries = [images] if isinstance(images, (str, os.PathLike)) else list(images)
    found = {}
    for entry in entries:
        path = Path(entry).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            if path == output_folder:
                raise ValueError('Input and output folders must differ.')
            candidates = path.rglob('*') if recursive else path.glob('*')
            candidates = (p for p in candidates if p.is_file() and
                          p.suffix.lower() in ('.tif', '.tiff') and
                          not _within(p.resolve(), output_folder))
        else:
            if path.suffix.lower() not in ('.tif', '.tiff'):
                raise ValueError(f'Expected a TIFF input: {path}')
            if _within(path, output_folder):
                raise ValueError('Do not use generated output files as raw inputs.')
            candidates = [path]
        for candidate in candidates:
            candidate = candidate.resolve()
            found[_source_id(candidate)] = candidate
    return sorted(found.values(), key=lambda p: os.path.normcase(str(p)))


def extract_sem_metadata(image_path):
    """Read common and namespaced fields using the registered metadata adapters."""
    return sem_metadata.read_metadata(image_path).csv_fields()


def _read_existing_csv(path):
    if not path.exists():
        return [], [], None
    digest = core._sha256(path)
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        if not columns or len(columns) != len(set(columns)) or 'source_id' not in columns:
            raise ValueError('Existing CSV is not a valid SEM batch CSV; it was not changed.')
        rows = list(reader)
    seen = set()
    for row in rows:
        if None in row or any(v is None for v in row.values()):
            raise ValueError('Existing CSV has malformed rows; it was not changed.')
        if not re.fullmatch('[0-9a-f]{64}', row['source_id']) or row['source_id'] in seen:
            raise ValueError('Existing CSV has missing/duplicate/invalid source IDs.')
        seen.add(row['source_id'])
    return rows, columns, digest


def _atomic_csv(path, rows, columns, expected_digest):
    """Replace a complete CSV atomically and refuse concurrent external edits."""
    handle = tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8-sig',
                                         dir=path.parent, prefix='.metadata_',
                                         suffix='.tmp', delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        current = core._sha256(path) if path.exists() else None
        if current != expected_digest:
            raise RuntimeError('CSV changed during processing; rerun to merge against the latest file.')
        os.replace(temporary, path)
    except PermissionError as exc:
        raise PermissionError('Cannot update metadata CSV. Close it in Excel and rerun.') from exc
    finally:
        temporary.unlink(missing_ok=True)


def _safe_output_file(root, relative):
    if not relative:
        return None
    path = (root / relative).resolve()
    return path if _within(path, root) else None


def _verified_report(output, source_digest, policy_hash):
    audit = audit_json_path(output)
    if not output.is_file() or not audit.is_file():
        return None
    try:
        report = json.loads(audit.read_text(encoding='utf-8'))
        if not isinstance(report, dict):
            return None
        if report.get('source_sha256') != source_digest or report.get('version') != core.VERSION:
            return None
        if report.get('processing', {}).get('policy_hash') != policy_hash:
            return None
        if report.get('output_sha256') != core._sha256(output):
            return None
        if not report.get('verification', {}).get('retained_pixels_identical'):
            return None
        if report.get('analysis_mask_file'):
            mask = _safe_output_file(output.parent, report['analysis_mask_file'])
            if mask is None or not mask.is_file() or core._sha256(mask) != report.get('analysis_mask_sha256'):
                return None
        if report.get('review_preview_file'):
            preview = _safe_output_file(output.parent, report['review_preview_file'])
            if preview is None or not preview.is_file():
                return None
        return report
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        return None


def _output_destination(root, source, identity, digest, policy_hash):
    stem = re.sub(r'[^\w.-]+', '_', source.stem, flags=re.UNICODE)[:65] or 'image'
    name = f'{stem}__{identity[:12]}__{digest[:12]}__{policy_hash[:12]}'
    for directory in ('images', 'review'):
        (root / directory).mkdir(exist_ok=True)
    number = 1
    while True:
        suffix = '' if number == 1 else f'_{number}'
        candidates = [root / directory / f'{name}{suffix}.tif' for directory in ('images', 'review')]
        for path in candidates:
            report = _verified_report(path, digest, policy_hash)
            if report is not None:
                return path, report
        if all(not p.exists() and not audit_json_path(p).exists() for p in candidates):
            return candidates[0], None
        number += 1


def _choice(mapping, source, identity, name_counts):
    if mapping is None:
        return None
    if not hasattr(mapping, 'items'):
        raise ValueError('crop_rows/exclusion_masks must be dictionaries keyed by source path, source_id or unique filename.')
    matched = []
    for key,value in mapping.items():
        key = str(key)
        if key == source.name:
            if name_counts[source.name] > 1:
                raise ValueError(f'Ambiguous filename {source.name}; use a full source path or source_id.')
            matched.append(value)
        elif key in (identity, str(source)) or Path(key).expanduser().resolve() == source:
            matched.append(value)
    if len(matched) > 1:
        raise ValueError('Multiple review decisions match this source. Provide only one key per image.')
    return matched[0] if matched else None


def _policy(source, manual_row, mask_value, allow_image_only_crop):
    mask = None
    mask_hash = ''
    if mask_value is not None:
        with Image.open(source) as im:
            shape = (im.height, im.width)
        mask = load_valid_mask(mask_value, shape)
        payload = str(mask.shape).encode() + np.packbits(mask).tobytes()
        mask_hash = hashlib.sha256(payload).hexdigest()
    if manual_row is not None and mask is not None:
        raise ValueError('Use either an approved crop row or an exclusion mask for an image, not both.')
    values = dict(batch_version=VERSION, crop_version=core.VERSION,
                  metadata_version=sem_metadata.VERSION,
                  registered_readers=[name for name,_,_ in sem_metadata._READERS],
                  manual_row=manual_row, mask_sha256=mask_hash,
                  allow_image_only_crop=bool(allow_image_only_crop))
    token = hashlib.sha256(json.dumps(values,sort_keys=True,default=core._json_value).encode()).hexdigest()
    return token, mask


def process_sem_images(images, output_folder, csv_name='metadata.csv', recursive=False,
                       *, crop_rows=None, exclusion_masks=None, allow_image_only_crop=False,
                       show_progress=True):
    """Process TIFFs incrementally with manufacturer-independent review handling.

    Unknown/unconfirmed footer boundaries generate byte-identical review copies,
    a candidate preview and needs_review CSV rows. Approve individual boundaries
    using crop_rows={"filename.tif": 768}, or supply exclusion_masks for overlays.
    True/nonzero mask pixels are INCLUDED. See README for support boundaries.
    """
    root = Path(output_folder).expanduser().resolve()
    if Path(csv_name).name != csv_name or Path(csv_name).suffix.lower() != '.csv':
        raise ValueError('csv_name must be a filename ending in .csv.')
    sources = _input_files(images, root, recursive)
    if not sources:
        raise ValueError('No input TIFFs found outside the output folder.')
    root.mkdir(parents=True, exist_ok=True)
    name_counts = Counter(p.name for p in sources)
    csv_path = root / csv_name
    lock = root / '.sem_batch.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError('Another run or a stale .sem_batch.lock exists. Check before retrying.') from exc
    with os.fdopen(fd, 'w') as f:
        f.write(str(os.getpid()))
    progress = None
    try:
        for directory in (root, root / 'images', root / 'review'):
            organise_json_records(directory)
        rows, columns, csv_digest = _read_existing_csv(csv_path)
        audit_paths_changed = False
        for existing in rows:
            relative = existing.get('output_image') or existing.get('review_image')
            image = _safe_output_file(root, relative)
            if image is not None and image.is_file():
                audit = audit_json_path(image)
                if audit.is_file():
                    relative_audit = audit.relative_to(root).as_posix()
                    if existing.get('audit_json') != relative_audit:
                        existing['audit_json'] = relative_audit
                        audit_paths_changed = True
        by_id = {row['source_id']: i for i, row in enumerate(rows)}
        summary = dict(input_count=len(sources), added=0, updated=0, skipped=0,
                       failed=0, needs_review=0, review_required=0,
                       csv_path=str(csv_path), errors=[])
        dirty = audit_paths_changed
        progress = Progress(len(sources), 'SEM extraction', enabled=show_progress).start()
        for source in progress.items(sources, name=lambda source: source.name):
            identity = _source_id(source)
            previous = rows[by_id[identity]] if identity in by_id else {}
            row = {k:v for k,v in previous.items() if k not in BASE_COLUMNS and
                   not k.startswith(('tiff.', 'tescan.', 'generic.'))}
            row.update(source_id=identity, source_file=source.name, source_path=str(source),
                       source_sha256='', status='failed', error='', review_required=True,
                       processed_utc=datetime.now(timezone.utc).isoformat(),
                       batch_version=VERSION, crop_version=core.VERSION)
            try:
                digest = core._sha256(source)
                row['source_sha256'] = digest
                manual = _choice(crop_rows, source, identity, name_counts)
                mask_value = _choice(exclusion_masks, source, identity, name_counts)
                policy_hash, mask = _policy(source, manual, mask_value, allow_image_only_crop)
                row['policy_hash'] = policy_hash
                old_role = 'review_image' if previous.get('status') == 'needs_review' else 'output_image'
                old_output = _safe_output_file(root, previous.get(old_role, ''))
                if (previous.get('status') in ('success','needs_review') and
                    previous.get('source_sha256') == digest and
                    previous.get('batch_version') == VERSION and
                    previous.get('policy_hash') == policy_hash and old_output and
                    _verified_report(old_output, digest, policy_hash) is not None):
                    summary['skipped'] += 1
                    summary['review_required'] += str(previous.get('review_required')).lower() == 'true'
                    continue
                info = sem_metadata.read_metadata(source)
                row.update(info.csv_fields())
                destination, report = _output_destination(root, source, identity, digest, policy_hash)
                if report is None:
                    report, destination = process_source(source, destination, info,
                        policy_hash=policy_hash, manual_row=manual, valid_mask=mask,
                        allow_image_only_crop=allow_image_only_crop)
                if report['source_sha256'] != digest or core._sha256(source) != digest:
                    raise RuntimeError('Input changed during processing; rerun after acquisition finishes.')
                processing = report['processing']
                status = processing['status']
                row.update(status=status, error='',
                           audit_json=audit_json_path(destination).relative_to(root).as_posix(),
                           dtype=report['dtype'], crop_method=processing['crop_method'],
                           review_required=processing['review_required'],
                           review_reason=processing['review_reason'],
                           proposed_crop_row=processing.get('proposed_crop_row'),
                           policy_hash=policy_hash)
                if status == 'success':
                    row.update(output_image=destination.relative_to(root).as_posix(),
                               output_sha256=report['output_sha256'],
                               cropped_width_px=report['cropped_shape'][1],
                               cropped_height_px=report['cropped_shape'][0],
                               removed_rows=report['removed_rows'])
                else:
                    row.update(review_image=destination.relative_to(root).as_posix(),
                               review_image_sha256=report['output_sha256'])
                    if report.get('review_preview_file'):
                        row['review_preview']=(destination.parent/report['review_preview_file']).relative_to(root).as_posix()
                    summary['needs_review'] += 1
                if report.get('analysis_mask_file'):
                    row['analysis_mask']=(destination.parent/report['analysis_mask_file']).relative_to(root).as_posix()
                    row['analysis_mask_sha256']=report['analysis_mask_sha256']
                    row['analysis_included_pixels']=report['analysis_included_pixels']
                summary['review_required'] += bool(processing['review_required'])
                summary['updated' if previous else 'added'] += 1
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
                summary['failed'] += 1
                summary['review_required'] += 1
                summary['errors'].append(dict(source_file=str(source), error=row['error']))
            if identity in by_id:
                rows[by_id[identity]] = row
            else:
                by_id[identity] = len(rows)
                rows.append(row)
            dirty = True
        progress.status('saving metadata CSV')
        if dirty or not csv_path.exists():
            columns = columns or list(BASE_COLUMNS)
            columns += sorted({k for row in rows for k in row} - set(columns))
            _atomic_csv(csv_path, rows, columns, csv_digest)
        return summary
    finally:
        if progress is not None:
            progress.close()
        lock.unlink(missing_ok=True)


from sem_paths import audit_json_path, organise_json_records
from sem_progress import Progress, instrument_functions
instrument_functions(globals(), exclude=('process_sem_images',))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', nargs='+', help='TIFF files and/or folders')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--csv-name', default='metadata.csv')
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--no-progress', action='store_true', help='Disable console progress')
    parser.add_argument('--allow-image-only-crop', action='store_true', help='Opt in only after validating the footer layout')
    parser.add_argument('--crop-rows', help='JSON mapping of source paths/unique filenames to approved crop rows')
    args = parser.parse_args()
    decisions = json.loads(Path(args.crop_rows).read_text(encoding='utf-8')) if args.crop_rows else None
    result = process_sem_images(args.images, args.output, args.csv_name, args.recursive,
        crop_rows=decisions, allow_image_only_crop=args.allow_image_only_crop,
        show_progress=not args.no_progress)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if result['failed'] else 0)
