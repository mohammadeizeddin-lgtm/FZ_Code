"""Batch SEM preprocessing using the user's preprocess_image parameter names.

Order: optional NLM -> optional Gaussian background subtraction -> optional
percentile clipping -> scaling to [0, 1]. Inputs are never overwritten.
"""
from os import path
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import argparse
import base64
import csv
import hashlib
import inspect
import json
import os
import tempfile

import numpy as np
try:
    import tifffile
except ImportError:
    tifffile = None  # Explicitly logged Pillow fallback when tifffile is unavailable.

VERSION = '1.0.0'
PREFIX = 'preprocess.'
PARAMETER_COLUMNS = [
    'image_file', 'processed_image', 'clipping', 'clipping_percentile_range',
    'gaussian_correction_filter', 'background_sigma', 'gentle_denoising',
]


def parse_path(path_str):
    """Expect detector_state_chemistry_image_ID.tif; not required by the batch."""
    name = Path(path_str).stem
    split = name.split('_')
    detector = split[0]
    state = split[1]
    chemistry = split[2]
    image_ID = int(split[3])
    return detector, state, chemistry, image_ID


def load_tiff(path_str):
    """Read a single grayscale TIFF; preserve its stored pixel intensities."""
    if not path.exists(path_str):
        raise FileNotFoundError(f'File not found: {path_str}')
    if tifffile is not None:
        with tifffile.TiffFile(path_str) as image_file:
            if len(image_file.pages) != 1:
                raise ValueError('Stacks/multi-page TIFFs need an explicit processing policy.')
            page = image_file.pages[0]
            if int(page.tags.get('Orientation').value if 'Orientation' in page.tags else 1) != 1:
                raise ValueError('Unsupported orientation; no silent reorientation is performed.')
            if int(page.photometric) != 1:
                raise ValueError('Expected black-is-zero grayscale TIFF.')
            return page.asarray()
    from PIL import Image
    with Image.open(path_str) as image_file:
        if image_file.format != 'TIFF' or image_file.n_frames != 1:
            raise ValueError('Expected a single-page TIFF.')
        if image_file.tag_v2.get(274, 1) != 1 or image_file.tag_v2.get(262, 1) != 1:
            raise ValueError('Unsupported orientation or grayscale interpretation.')
        return np.asarray(image_file).copy()


def preprocess_image(image, clipping_percentile_range=None, *,
                     gaussian_correction_filter=False, background_sigma=None,
                     gentle_denoising=False, return_details=False):
    """Return float32 in [0,1]; existing parameter names and defaults retained.

    return_details=True additionally returns scalar processing details for logging.
    Noise sigma/h use SOURCE intensity units; background_sigma is spatial pixels.
    This is not mask-aware. Remove information bars and use a valid rectangular
    specimen region. Denoising/noise estimates and background scales need feature
    validation; real fine texture or broad material gradients can be suppressed.
    """
    image = np.asarray(image)
    if image.ndim != 2 or image.size == 0:
        raise ValueError('Expected a non-empty 2D grayscale image.')
    if np.iscomplexobj(image):
        raise ValueError('Expected real-valued image intensities.')
    print(f'Image shape: {image.shape}, Image dtype: {image.dtype}')
    image_float = image.astype(np.float32)
    if not np.isfinite(image_float).all():
        raise ValueError('Image contains NaN, infinity, or values outside float32 range.')
    for name, flag in [('gaussian_correction_filter', gaussian_correction_filter),
                       ('gentle_denoising', gentle_denoising), ('return_details', return_details)]:
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f'{name} must be True or False.')

    bounds = None
    if clipping_percentile_range is not None:
        try:
            bounds = np.asarray(clipping_percentile_range, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError('Provide None or two percentile numbers.') from exc
        if (bounds.shape != (2,) or not np.isfinite(bounds).all()
                or not 0 <= bounds[0] < bounds[1] <= 100):
            raise ValueError('Percentiles must satisfy 0 <= lower < upper <= 100.')
    if gaussian_correction_filter:
        if (background_sigma is None or isinstance(background_sigma, (bool, np.bool_))
                or not isinstance(background_sigma, (int, float, np.integer, np.floating))
                or not np.isfinite(background_sigma) or background_sigma <= 0):
            raise ValueError('Provide a positive background_sigma in pixels.')

    details = dict(
        input_dtype=str(image.dtype), width_px=image.shape[1], height_px=image.shape[0],
        input_min=float(image_float.min()), input_max=float(image_float.max()),
        gentle_denoising_requested=bool(gentle_denoising), denoising_applied=False,
        noise_sigma_source_units=None, nlm_h_source_units=None,
        nlm_h_factor=0.6, nlm_patch_size=5, nlm_patch_distance=6, nlm_fast_mode=True,
        denoising_skip_reason='',
        gaussian_correction_filter=bool(gaussian_correction_filter),
        background_sigma_px=float(background_sigma) if gaussian_correction_filter else None,
        background_min=None, background_max=None, background_mean=None,
        negative_pixels_after_background=None,
        clipping_requested=bounds is not None,
        clipping_percentile_low=None if bounds is None else float(bounds[0]),
        clipping_percentile_high=None if bounds is None else float(bounds[1]),
        pixels_clipped_low=0, pixels_clipped_high=0,
        processing_order=[],
    )
    # Optional NLM denoising. This matches the user's raw-float intensity scale.
    if gentle_denoising:
        from skimage.restoration import denoise_nl_means, estimate_sigma
        noise_sigma = float(estimate_sigma(image_float, channel_axis=None))
        if not np.isfinite(noise_sigma) or noise_sigma < 0:
            raise ValueError('Could not obtain a valid noise estimate.')
        details['noise_sigma_source_units'] = noise_sigma
        if noise_sigma > 0:
            image_float = denoise_nl_means(
                image_float, h=0.6 * noise_sigma, sigma=noise_sigma,
                patch_size=5, patch_distance=6, fast_mode=True,
                preserve_range=True, channel_axis=None,
            ).astype(np.float32)
            if not np.isfinite(image_float).all():
                raise ValueError('Denoising produced non-finite intensity values.')
            details['denoising_applied'] = True
            details['nlm_h_source_units'] = 0.6 * noise_sigma
            details['processing_order'].append('non_local_means')
        else:
            details['denoising_skip_reason'] = 'Estimated noise sigma is zero.'
    if gaussian_correction_filter:
        from scipy.ndimage import gaussian_filter
        background = gaussian_filter(image_float, sigma=background_sigma, mode='reflect')
        details.update(background_min=float(background.min()),
                       background_max=float(background.max()), background_mean=float(background.mean()))
        image_float = image_float - background
        details['negative_pixels_after_background'] = int(np.count_nonzero(image_float < 0))
        details['processing_order'].append('gaussian_background_subtraction')
    if clipping_percentile_range is None:
        low, high = image_float.min(), image_float.max()
    else:
        low, high = np.percentile(image_float, bounds)
    if high <= low:
        raise ValueError('Selected intensity limits have no variation; cannot normalize.')
    details['scaling_low'] = float(low)
    details['scaling_high'] = float(high)
    if clipping_percentile_range is not None:
        details['pixels_clipped_low'] = int(np.count_nonzero(image_float < low))
        details['pixels_clipped_high'] = int(np.count_nonzero(image_float > high))
        image_float = np.clip(image_float, low, high)
        details['processing_order'].append('percentile_clipping')
    normalised_image = ((image_float - low) / (high - low)).astype(np.float32, copy=False)
    if not np.isfinite(normalised_image).all():
        raise ValueError('Normalisation produced non-finite values.')
    details.update(output_dtype=str(normalised_image.dtype),
                   output_min=float(normalised_image.min()), output_max=float(normalised_image.max()))
    details['processing_order'].append('normalisation_0_1')
    return (normalised_image, details) if return_details else normalised_image


def save_tiff(image, path_str, *, metadata=None, resolution_tags=None):
    """Save a derived float32 TIFF. Optional metadata describes its provenance.

    Original private vendor blocks are archived in JSON, not blindly embedded
    into a TIFF whose intensity representation has changed from integer to float.
    """
    if tifffile is not None:
        options = dict(photometric='minisblack', metadata=metadata)
        if resolution_tags and 282 in resolution_tags and 283 in resolution_tags:
            def rational(value):
                if isinstance(value,(tuple,list)) and len(value)==2:
                    return float(value[0])/float(value[1])
                return float(value)
            options.update(resolution=(rational(resolution_tags[282]), rational(resolution_tags[283])),
                           resolutionunit=int(resolution_tags.get(296, 1)))
        tifffile.imwrite(path_str, image.astype(np.float32), **options)
    else:
        from PIL import Image, TiffImagePlugin
        info = TiffImagePlugin.ImageFileDirectory_v2()
        if metadata is not None:
            info[270] = json.dumps(_json_safe(metadata), ensure_ascii=True)
        for code in (282, 283, 296):
            if resolution_tags and code in resolution_tags:
                info[code] = resolution_tags[code]
        Image.fromarray(image.astype(np.float32)).save(path_str, format='TIFF', compression='raw', tiffinfo=info)


def _json_safe(value):
    if isinstance(value, bytes):
        return dict(encoding='base64', value=base64.b64encode(value).decode('ascii'))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, 'numerator') and hasattr(value, 'denominator') and not isinstance(value, (int, float)):
        return dict(numerator=value.numerator, denominator=value.denominator)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _hash_file(file):
    result = hashlib.sha256()
    with open(file, 'rb') as f:
        for chunk in iter(lambda:f.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _hash_json(value):
    return hashlib.sha256(json.dumps(_json_safe(value),sort_keys=True).encode()).hexdigest()


def _versions():
    result = {}
    for package in ('numpy', 'scipy', 'scikit-image', 'Pillow', 'tifffile'):
        try:result[package] = version(package)
        except PackageNotFoundError:result[package] = 'not_installed'
    result['io_backend'] = 'tifffile' if tifffile is not None else 'Pillow'
    return result


def _inside(file, folder):
    try:file.relative_to(folder);return True
    except ValueError:return False


def _read_table(file, required):
    with open(file, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)) or not set(required) <= set(fields):
            raise ValueError(f'CSV has an unrecognised schema: {file}')
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f'CSV contains malformed rows: {file}')
    return fields, rows


def _manifest(file, scope=None):
    """Use only current approved outputs; never process masks/review/history TIFFs."""
    _, rows = _read_table(file, ('status', 'output_image', 'source_id'))
    root = file.parent.resolve()
    selected = []
    for row in rows:
        if row['status'] != 'success' or str(row.get('review_required','')).lower() in ('true','1','yes'):
            continue
        relative = row.get('output_image', '')
        if not relative:continue
        image_file = (root / relative).resolve()
        if not _inside(image_file, root):
            raise ValueError('Input metadata CSV refers to an image outside its output folder.')
        if scope is not None and image_file != scope and not _inside(image_file, scope):continue
        selected.append((image_file, dict(row), str(file)))
    return selected


def _discover(images, output_folder, recursive, input_metadata_csv):
    entries = [images] if isinstance(images, (str, os.PathLike)) else list(images)
    selected = {}
    explicit_manifest = Path(input_metadata_csv).expanduser().resolve() if input_metadata_csv else None
    for entry in entries:
        file = Path(entry).expanduser().resolve()
        if not file.exists():raise FileNotFoundError(file)
        if _inside(file, output_folder):
            raise ValueError('Choose an output folder separate from the input files/folder.')
        manifest = explicit_manifest
        if manifest is None:
            candidates = [file/'metadata.csv'] if file.is_dir() else []
            if file.name == 'images' or file.is_file():candidates.append(file.parent/'metadata.csv')
            if file.is_file() and file.parent.name == 'images':candidates.append(file.parent.parent/'metadata.csv')
            manifest = next((p for p in candidates if p.is_file()), None)
        if manifest is not None:
            selection = _manifest(manifest, scope=file)
        else:
            iterator = file.rglob('*') if file.is_dir() and recursive else (file.glob('*') if file.is_dir() else [file])
            selection = [(p.resolve(), {}, '') for p in iterator if p.is_file()
                and p.suffix.lower() in ('.tif','.tiff') and not _inside(p.resolve(), output_folder)
                and not (file.is_dir() and inside_generated_preprocessing(p, file))
                and not p.name.lower().endswith(('.analysis_mask.tif','.mask.tif'))]
        for image_file, metadata, manifest_name in selection:
            if _inside(image_file, output_folder):continue
            selected[os.path.normcase(str(image_file))] = (image_file, metadata, manifest_name)
    return sorted(selected.values(), key=lambda item:os.path.normcase(str(item[0])))


def _tiff_metadata(file):
    if tifffile is not None:
        with tifffile.TiffFile(file) as t:
            tags = {tag.code:tag.value for tag in t.pages[0].tags.values()}
    else:
        from PIL import Image
        with Image.open(file) as t:tags = dict(t.tag_v2)
    return tags


def _source_audit_metadata(file, metadata):
    """Supplement source CSV fields from the extraction audit, if available."""
    audit = source_audit_path(file)
    if not audit.exists():return metadata
    try:data=json.loads(audit.read_text(encoding='utf-8'))
    except (OSError, ValueError):return metadata
    if not isinstance(data,dict):return metadata
    result = dict(data.get('source_metadata',{})) if isinstance(data.get('source_metadata'),dict) else {}
    if isinstance(data.get('readable_metadata'),dict):result.update(data['readable_metadata'])
    calibration=data.get('calibration',{})
    if not isinstance(calibration,dict):calibration={}
    for axis in ('x','y'):
        value=calibration.get(f'PixelSize{axis.upper()}_metres')
        if isinstance(value,(int,float)) and np.isfinite(value) and value>0:
            result[f'pixel_size_{axis}_nm']=value*1e9
    result.update(metadata)
    return result


def _atomic_csv(file, rows, fields, expected_hash):
    handle = tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8-sig',
                                         dir=file.parent, prefix='.preprocessing_', suffix='.tmp', delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            writer=csv.DictWriter(handle,fieldnames=fields)
            writer.writeheader();writer.writerows(rows)
            handle.flush();os.fsync(handle.fileno())
        current = _hash_file(file) if file.exists() else None
        if current != expected_hash:
            raise RuntimeError('Preprocessing CSV changed during this run; rerun to merge the latest version.')
        os.replace(temporary,file)
    except PermissionError as exc:
        raise PermissionError('Close the preprocessing CSV in Excel, then rerun.') from exc
    finally:temporary.unlink(missing_ok=True)


def _scalar_csv(value):
    if value is None:return ''
    if isinstance(value,(dict,list,tuple)):
        return json.dumps(_json_safe(value),ensure_ascii=True)
    return _json_safe(value)


def _csv_text(value):
    value = _scalar_csv(value)
    return '' if value is None else str(value)


def _parameter_rows(records):
    """Publish acquisition fields plus only user-selected preprocessing values."""
    result=[]
    for record in records:
        # Failure/review diagnostics belong to JSON and the returned summary.
        if record.get(PREFIX+'status')!='success':continue
        settings=record.get(PREFIX+'settings',{})
        if isinstance(settings,str):settings=json.loads(settings)
        if not isinstance(settings,dict):raise ValueError('Invalid stored preprocessing settings.')
        bounds=settings.get('clipping_percentile_range')
        gaussian=bool(settings.get('gaussian_correction_filter',False))
        row={k:_csv_text(v) for k,v in record.items()
             if not k.startswith(PREFIX) and k not in PARAMETER_COLUMNS}
        row.update(
            image_file=record.get(PREFIX+'input_file',''),
            processed_image=record.get(PREFIX+'output_image',''),
            clipping=str(bounds is not None),
            clipping_percentile_range='None' if bounds is None else str(tuple(bounds)),
            gaussian_correction_filter=str(gaussian),
            background_sigma=_csv_text(settings.get('background_sigma')) if gaussian else 'None',
            gentle_denoising=str(bool(settings.get('gentle_denoising',False))),
        )
        result.append(row)
    return result


def _read_history(records_file,csv_file):
    """Migrate the old detailed CSV to JSON without reprocessing any images."""
    csv_fields,csv_rows=([],[])
    if csv_file.exists():csv_fields,csv_rows=_read_table(csv_file,())
    if csv_fields and PREFIX+'record_id' not in csv_fields and not set(PARAMETER_COLUMNS)<=set(csv_fields):
        raise ValueError('Unrecognised preprocessing CSV; nothing was overwritten.')
    migrated=False
    if records_file.exists():
        data=json.loads(records_file.read_text(encoding='utf-8'))
        if not isinstance(data,dict) or data.get('schema')!=2 or not isinstance(data.get('records'),list):
            raise ValueError('Unrecognised preprocessing history JSON; nothing was overwritten.')
        records=data['records']
    elif not csv_rows:
        records=[]
    elif PREFIX+'record_id' in csv_fields:
        records=csv_rows
        migrated=True
    elif set(PARAMETER_COLUMNS)<=set(csv_fields):
        records=[]
        for public in csv_rows:
            image=(csv_file.parent/public['processed_image']).resolve()
            if not _inside(image,csv_file.parent.resolve()):
                raise ValueError('CSV output path is outside the output folder.')
            audit=json.loads(audit_json_path(image).read_text(encoding='utf-8'))
            record={k:v for k,v in public.items() if k not in PARAMETER_COLUMNS}
            record.update({PREFIX+'record_id':audit['record_id'],PREFIX+'status':'success',
                PREFIX+'input_file':public['image_file'],PREFIX+'output_image':public['processed_image'],
                PREFIX+'settings':json.dumps(audit['settings']),
                PREFIX+'output_sha256':audit['output_sha256']})
            records.append(record)
        migrated=True
    else:
        raise ValueError('Unrecognised preprocessing CSV; nothing was overwritten.')

    # Retain user-added notes. Acquisition fields and parameter fields are managed.
    by_output={r.get(PREFIX+'output_image'):r for r in records if r.get(PREFIX+'output_image')}
    for relative, record in by_output.items():
        manifest = record.get(PREFIX+'input_metadata_csv')
        upstream_audit = record.get('audit_json')
        if manifest and upstream_audit:
            manifest_folder = Path(manifest).parent
            upstream_path = (manifest_folder / upstream_audit).resolve()
            if _inside(upstream_path, manifest_folder.resolve()):
                moved = json_record_path(upstream_path)
                if moved.is_file():
                    reference = moved.relative_to(manifest_folder).as_posix()
                    if upstream_audit != reference:
                        record['audit_json'] = reference
                        migrated = True
        image = (csv_file.parent / relative).resolve()
        if _inside(image, csv_file.parent.resolve()) and image.is_file():
            audit = audit_json_path(image)
            if audit.is_file():
                reference = audit.relative_to(csv_file.parent).as_posix()
                if record.get(PREFIX+'audit_json') != reference:
                    record[PREFIX+'audit_json'] = reference
                    migrated = True
    if set(PARAMETER_COLUMNS)<=set(csv_fields):
        for public in csv_rows:
            record=by_output.get(public['processed_image'])
            if record is None:raise ValueError('CSV row has no matching history record; nothing was overwritten.')
            for key,value in public.items():
                if key not in PARAMETER_COLUMNS and not key.startswith(PREFIX) and key not in record:
                    record[key]=value
                    migrated=True
                elif key not in PARAMETER_COLUMNS and key in record and key.endswith('_notes'):
                    if record[key]!=value:record[key]=value;migrated=True
    return records,csv_fields,csv_rows,migrated


def _atomic_history(file,records,expected_hash):
    handle=tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=file.parent,
        prefix='.preprocessing_records_',suffix='.tmp',delete=False)
    temporary=Path(handle.name)
    try:
        with handle:
            json.dump(dict(schema=2,records=_json_safe(records)),handle,indent=2,ensure_ascii=True)
            handle.flush();os.fsync(handle.fileno())
        current=_hash_file(file) if file.exists() else None
        if current!=expected_hash:raise RuntimeError('Preprocessing history changed during this run.')
        os.replace(temporary,file)
    finally:temporary.unlink(missing_ok=True)


def _existing_result(root, row, record_id):
    relative = row.get(PREFIX+'output_image','')
    if not relative:return None
    file = (root / relative).resolve()
    if not _inside(file, root) or not file.is_file():return None
    audit_file=audit_json_path(file)
    if not audit_file.is_file():return None
    try:
        audit=json.loads(audit_file.read_text())
        if audit.get('record_id') != record_id or _hash_file(file) != audit.get('output_sha256'):return None
        return audit
    except (OSError,ValueError,AttributeError):return None


def preprocess_images(images, output_folder=None, clipping_percentile_range=None, *,
                      gaussian_correction_filter=False, background_sigma=None,
                      gentle_denoising=False, recursive=False,
                      csv_name='preprocessing_metadata.csv', input_metadata_csv=None,
                      show_progress=True):
    """Preprocess approved TIFF outputs and maintain a separate extended CSV.

    Accept one file, a list of files/folders, or a folder. When extraction
    metadata.csv exists, use its active approved image outputs automatically,
    irrespective of the images/ subdirectory. recursive controls plain-folder
    discovery, not explicitly listed manifest files. Masks are flagged for review
    because the user's preprocess_image is not mask-aware.

    Each unique source path + source content + recipe has one record. New images
    or different recipes APPEND history. Identical reruns skip verified results.
    Existing files and original extraction CSVs are never overwritten.
    Omit output_folder for <sample>_preprocessed[_clipped] inside the input
    folder; explicit output paths are honoured. Image filenames are unchanged.
    JSON records are kept in json/; legacy records are migrated on use.
    """
    if not isinstance(images,(str,os.PathLike)):images=list(images)
    root=(preprocessing_output_folder(images,clipping_percentile_range) if output_folder is None
          else Path(output_folder).expanduser().resolve())
    if Path(csv_name).name!=csv_name or Path(csv_name).suffix.lower()!='.csv':
        raise ValueError('csv_name must be a CSV filename.')
    selected=_discover(images,root,recursive,input_metadata_csv)
    if not selected:raise ValueError('No approved TIFF image inputs found. Check folder, CSV status and recursive setting.')
    csv_file=root/csv_name
    if any(manifest and Path(manifest).resolve()==csv_file for _,_,manifest in selected):
        raise ValueError('The preprocessing CSV must be separate from the extraction metadata CSV.')
    root.mkdir(parents=True,exist_ok=True)
    lock=root/'.preprocessing.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as exc:raise RuntimeError('Another run or a stale .preprocessing.lock exists.') from exc
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    progress=None
    try:
        organise_json_records(root)
        csv_hash=_hash_file(csv_file) if csv_file.exists() else None
        records_file=json_record_path(csv_file.with_suffix('.records.json'))
        records_hash=_hash_file(records_file) if records_file.exists() else None
        rows,fields,public_before,migrated=_read_history(records_file,csv_file)
        indices={}
        for i,row in enumerate(rows):
            key=row[PREFIX+'record_id']
            if not key or key in indices:raise ValueError('Existing preprocessing CSV has missing/duplicate record IDs.')
            indices[key]=i
        settings=dict(clipping_percentile_range=_json_safe(clipping_percentile_range),
                      gaussian_correction_filter=gaussian_correction_filter,
                      background_sigma=background_sigma if gaussian_correction_filter else None,
                      gentle_denoising=gentle_denoising,
                      nlm_h_factor=0.6,nlm_patch_size=5,nlm_patch_distance=6,nlm_fast_mode=True)
        libraries=_versions()
        implementation_hash=_hash_json(inspect.getsource(preprocess_image)+inspect.getsource(save_tiff))
        recipe_hash=_hash_json(dict(version=VERSION,settings=settings,libraries=libraries,implementation_hash=implementation_hash))
        summary=dict(input_count=len(selected),processed=0,skipped=0,failed=0,needs_review=0,
                     output_folder=str(root),csv_path=str(csv_file),errors=[])
        dirty=migrated
        progress=Progress(len(selected),'Preprocessing',enabled=show_progress).start()
        for source,source_metadata,manifest in progress.items(selected,name=lambda item:item[0].name):
            source_id=hashlib.sha256(os.path.normcase(str(source)).encode()).hexdigest()
            digest=''
            try:digest=_hash_file(source)
            except OSError:pass
            metadata=_source_audit_metadata(source,source_metadata)
            # An upstream CSV can add blank columns for older images. That does
            # not change their scientific metadata or require new processing.
            metadata_signature={k:str(_scalar_csv(v)) for k,v in metadata.items()
                                if v is not None and str(_scalar_csv(v))!=''}
            if 'audit_json' in metadata_signature:
                metadata_signature['audit_json']=legacy_json_reference(metadata_signature['audit_json'])
            metadata_hash=_hash_json(metadata_signature)
            record_id=_hash_json([source_id,digest or 'unreadable',recipe_hash,metadata_hash])
            previous=rows[indices[record_id]] if record_id in indices else {}
            row=dict(previous)
            row.update(metadata)
            # All new processing fields use a prefix, so source acquisition fields survive.
            for name in list(row):
                if name.startswith(PREFIX):row.pop(name)
            row.update({PREFIX+k:_scalar_csv(v) for k,v in dict(
                record_id=record_id,version=VERSION,input_file=source.name,input_path=str(source),
                input_sha256=digest,recipe_sha256=recipe_hash,implementation_sha256=implementation_hash,
                input_metadata_csv=manifest,input_metadata_sha256=metadata_hash,
                settings=settings,libraries=libraries,
                gentle_denoising_requested=gentle_denoising,
                gaussian_correction_filter=gaussian_correction_filter,
                clipping_percentile_range=clipping_percentile_range,
                status='failed',error='',timestamp_utc=datetime.now(timezone.utc).isoformat()).items()})
            try:
                if not digest:digest=_hash_file(source)
                if source_metadata.get('analysis_mask'):
                    row[PREFIX+'status']='needs_review'
                    row[PREFIX+'error']='Source has an analysis mask. This preprocess_image is not mask-aware; no filtering or scaling was applied.'
                    summary['needs_review']+=1
                else:
                    audit=_existing_result(root,previous,record_id)
                    if audit is not None:
                        summary['skipped']+=1
                        continue
                    # Recover a completed file/audit pair if a previous run lost its CSV.
                    stem=source.stem[:65]+'__preprocessed__'+record_id[:16]
                    number=1
                    while True:
                        suffix='' if number==1 else f'_{number}'
                        destination=root/f'{stem}{suffix}.tif'
                        candidate={PREFIX+'output_image':destination.name}
                        audit=_existing_result(root,candidate,record_id)
                        if audit is not None:break
                        if not destination.exists() and not audit_json_path(destination).exists():break
                        number+=1
                    if audit is None:
                        image=load_tiff(str(source))
                        processed_image,details=preprocess_image(image,clipping_percentile_range,
                            gaussian_correction_filter=gaussian_correction_filter,
                            background_sigma=background_sigma,gentle_denoising=gentle_denoising,
                            return_details=True)
                        source_tags=_tiff_metadata(source)
                        calibration={k:v for k,v in metadata.items() if k in ('pixel_size_x_nm','pixel_size_y_nm','calibration_status')}
                        description=dict(axes='YX',source_file=source.name,source_sha256=digest,
                            preprocessing_record_id=record_id,preprocessing_recipe=recipe_hash,
                            calibration=calibration,processing=_json_safe(details))
                        created=False
                        try:
                            # Reserve the path before writing; never overwrite existing data.
                            with open(destination,'xb'):created=True
                            save_tiff(processed_image,str(destination),metadata=description,resolution_tags=source_tags)
                            check=load_tiff(str(destination))
                            if check.dtype!=np.float32 or not np.array_equal(check,processed_image):
                                raise RuntimeError('Saved TIFF failed exact float32 pixel verification.')
                            if _hash_file(source)!=digest:raise RuntimeError('Input changed while processing; rerun once acquisition is finished.')
                            audit=dict(record_id=record_id,source_sha256=digest,output_sha256=_hash_file(destination),
                                source_metadata=_json_safe(metadata),source_tiff_tags=_json_safe(source_tags),
                                settings=_json_safe(settings),details=_json_safe(details),libraries=libraries,
                                verification=dict(float32_pixels_match=True,input_unchanged=True,shape_unchanged=True))
                            with open(audit_json_path(destination),'x',encoding='utf-8') as f:
                                json.dump(audit,f,indent=2,ensure_ascii=True)
                        except Exception:
                            if created:destination.unlink(missing_ok=True)
                            raise
                    if _hash_file(source)!=digest:raise RuntimeError('Source changed during processing.')
                    row.update({PREFIX+k:_scalar_csv(v) for k,v in audit['details'].items()})
                    row.update({PREFIX+'status':'success',PREFIX+'error':'',
                                PREFIX+'output_image':destination.name,PREFIX+'audit_json':audit_json_path(destination).relative_to(root).as_posix(),
                                PREFIX+'output_sha256':audit['output_sha256']})
                    summary['processed']+=1
            except Exception as exc:
                row[PREFIX+'error']=f'{type(exc).__name__}: {exc}'
                summary['failed']+=1
                summary['errors'].append(dict(input_file=str(source),error=row[PREFIX+'error']))
            if record_id in indices:rows[indices[record_id]]=row
            else:indices[record_id]=len(rows);rows.append(row)
            dirty=True
        progress.status('saving preprocessing CSV')
        public_rows=_parameter_rows(rows)
        # Keep acquisition/custom columns, then append the seven simple columns.
        public_fields=[k for k in fields if not k.startswith(PREFIX) and k not in PARAMETER_COLUMNS]
        public_fields+=sorted({k for row in public_rows for k in row}
                              -set(public_fields)-set(PARAMETER_COLUMNS))
        public_fields+=PARAMETER_COLUMNS
        canonical=[{k:row.get(k,'') for k in public_fields} for row in public_rows]
        if dirty or not records_file.exists():
            _atomic_history(records_file,rows,records_hash)
        if not csv_file.exists() or fields!=public_fields or public_before!=canonical:
            _atomic_csv(csv_file,canonical,public_fields,csv_hash)
        return summary
    finally:
        if progress is not None:progress.close()
        lock.unlink(missing_ok=True)


from sem_paths import (audit_json_path, source_audit_path, json_record_path,
                       organise_json_records, legacy_json_reference, preprocessing_output_folder,
                       inside_generated_preprocessing)
from sem_progress import Progress, instrument_functions
instrument_functions(globals(),exclude=('preprocess_images',))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images',nargs='+')
    parser.add_argument('--output',help='Optional output folder; default: <sample>_preprocessed[_clipped] inside the input folder')
    parser.add_argument('--recursive',action='store_true')
    parser.add_argument('--no-progress',action='store_true',help='Disable console progress')
    parser.add_argument('--gentle-denoising',action='store_true')
    parser.add_argument('--gaussian-correction-filter',action='store_true')
    parser.add_argument('--background-sigma',type=float)
    parser.add_argument('--clip',nargs=2,type=float,metavar=('LOW','HIGH'))
    parser.add_argument('--input-metadata-csv')
    args=parser.parse_args()
    result=preprocess_images(args.images,args.output,args.clip,
        gaussian_correction_filter=args.gaussian_correction_filter,background_sigma=args.background_sigma,
        gentle_denoising=args.gentle_denoising,recursive=args.recursive,input_metadata_csv=args.input_metadata_csv,
        show_progress=not args.no_progress)
    print(json.dumps(result,indent=2))
    raise SystemExit(1 if result['failed'] else 0)
