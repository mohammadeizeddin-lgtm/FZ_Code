"""Output-folder naming and backwards-compatible JSON record locations."""
from pathlib import Path
import hashlib
import json
import os
import re


def json_record_path(legacy_path):
    """Store a record in its parent's json/ folder; migrate an adjacent record.

    Moves retain exact bytes. A conflicting destination is never overwritten.
    Existing identical copies are deduplicated after verification.
    """
    old = Path(legacy_path)
    if old.suffix.lower() != '.json':
        raise ValueError('Expected a JSON record path.')
    if old.parent.name == 'json':
        old.parent.mkdir(parents=True, exist_ok=True)
        return old
    target = old.parent / 'json' / old.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if old.is_file():
        if target.exists():
            def digest(file):
                h = hashlib.sha256()
                with open(file, 'rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        h.update(chunk)
                return h.digest()
            if not target.is_file() or digest(old) != digest(target):
                raise FileExistsError(f'Conflicting JSON records: {old} and {target}. Both were retained.')
            old.unlink()
        else:
            old.rename(target)
    return target


def audit_json_path(image):
    return json_record_path(Path(image).with_suffix('.json'))


def source_audit_path(image):
    """Read arbitrary source sidecars without moving the user's original files."""
    old = Path(image).with_suffix('.json')
    new = old.parent / 'json' / old.name
    return new if new.is_file() else old


def organise_json_records(folder):
    """Move recognised generated records only; leave unrelated JSON untouched."""
    folder = Path(folder)
    if not folder.is_dir():
        return
    for file in sorted(folder.glob('*.json')):
        try:
            record = json.loads(file.read_text(encoding='utf-8'))
        except (OSError, ValueError, UnicodeError):
            continue
        if isinstance(record, dict) and (
                'measurement_id' in record or 'record_id' in record
                or ('source_sha256' in record and 'output_sha256' in record)
                or ('schema' in record and 'records' in record)):
            json_record_path(file)


def legacy_json_reference(value):
    """Normalise a storage-only audit relocation for existing metadata hashes."""
    file = Path(str(value))
    return (file.parent.parent / file.name).as_posix() if file.parent.name == 'json' else value


def preprocessing_output_folder(images, clipping_percentile_range):
    """Infer a sample-named child folder; explicit output_folder stays untouched."""
    entries = [images] if isinstance(images, (str, os.PathLike)) else list(images)
    if not entries:
        raise ValueError('Provide at least one input image or folder.')
    scopes = []
    for entry in entries:
        file = Path(entry).expanduser().resolve()
        scope = file if file.is_dir() else file.parent
        if scope.name == 'images' and (scope.parent / 'metadata.csv').is_file():
            scope = scope.parent
        scopes.append(scope)
    parent = Path(os.path.commonpath([str(p) for p in scopes]))
    sample = parent
    while sample.name.lower() in ('images', 'raw', 'processed', 'preprocessed', 'segmented') and sample.parent != sample:
        sample = sample.parent
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', '_', sample.name).strip(' .') or 'SEM'
    suffix = '_preprocessed_clipped' if clipping_percentile_range is not None else '_preprocessed'
    return parent / (name + suffix)


def inside_generated_preprocessing(file, scope):
    """Do not ingest earlier sibling preprocessing runs during raw recursion."""
    scope = Path(scope)
    for parent in Path(file).parents:
        if parent == scope or parent == parent.parent:
            break
        if ((parent / 'preprocessing_metadata.csv').is_file()
                or (parent / 'json' / 'preprocessing_metadata.records.json').is_file()):
            return True
    return False
