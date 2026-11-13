"""Conservative flat-background SEM footer detection and lossless TIFF export.

Dependencies: numpy, Pillow. Run on raw integer images before normalisation.
See README.md for supported layouts and TIFF export limitations.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import re
import struct

import numpy as np
import PIL
from PIL import Image, TiffImagePlugin

VERSION = "2.0.0"
DETECTOR_PARAMETERS = dict(search_bottom_fraction=0.35, border_dominance=0.995,
                           footer_median_dominance=0.80, footer_row_dominance=0.65,
                           footer_fraction_passing=0.90, background_consistency=0.80,
                           preceding_dominance_max=0.50)


def _detect_information_bar(image):
    """Find one flat-border/flat-background footer, or raise rather than guess."""
    a = np.asarray(image)
    if a.ndim != 2 or a.dtype.kind != "u" or a.dtype.itemsize not in (1, 2):
        raise ValueError("Expected a raw 2D uint8 or uint16 grayscale image.")
    h, w = a.shape
    if h < 32 or w < 64:
        raise ValueError("Image is too small for this detector.")
    start = int(h * (1 - DETECTOR_PARAMETERS['search_bottom_fraction']))
    window = max(8, int(h * 0.01))
    minimum_footer = max(8, int(h * 0.01))
    dominance = np.zeros(h, dtype=float)
    modes = np.zeros(h, dtype=a.dtype)
    for y in range(max(0, start - window), h):
        values, counts = np.unique(a[y], return_counts=True)
        i = counts.argmax()
        dominance[y], modes[y] = counts[i] / w, values[i]

    candidates = []
    for y in range(start, h - minimum_footer + 1):
        if dominance[y] < DETECTOR_PARAMETERS['border_dominance']:
            continue
        if np.median(dominance[y-window:y]) >= DETECTOR_PARAMETERS['preceding_dominance_max']:
            continue
        tail = dominance[y:]
        if np.median(tail) < DETECTOR_PARAMETERS['footer_median_dominance']:
            continue
        if np.mean(tail >= DETECTOR_PARAMETERS['footer_row_dominance']) < DETECTOR_PARAMETERS['footer_fraction_passing']:
            continue
        _, counts = np.unique(modes[y:], return_counts=True)
        if counts.max() / len(tail) < DETECTOR_PARAMETERS['background_consistency']:
            continue
        # An entirely flat band is not sufficient evidence of an information bar.
        if not np.any(tail < 0.98):
            continue
        candidates.append(y)

    # Several adjacent blank rows can describe the same boundary; use its top.
    groups = []
    for y in candidates:
        if not groups or y - groups[-1][-1] > window:
            groups.append([y])
        else:
            groups[-1].append(y)
    if len(groups) != 1:
        raise ValueError("No unique supported information-bar boundary found; review manually.")
    return groups[0][0]


def remove_information_bar(image):
    """Return a copy cropped above the detected bottom information bar.

    Preserves dtype and retained pixel values; never normalises, clips or resizes.
    This is a conservative heuristic for separate flat-background bottom footers,
    not a general text/scale-bar detector. Raises ValueError for unsupported or
    ambiguous layouts. Review outputs before using on a new acquisition layout.
    """
    crop_row = _detect_information_bar(image)
    return np.asarray(image)[:crop_row, :].copy()


def _tescan_blocks(blob):
    """TESCAN records: little-endian size (excluding itself), type, payload."""
    out, pos = [], 0
    while pos < len(blob):
        if pos + 6 > len(blob):
            raise ValueError("Malformed TESCAN metadata record.")
        size, kind = struct.unpack_from('<IH', blob, pos)
        end = pos + 4 + size
        if size < 2 or end > len(blob):
            raise ValueError("Malformed TESCAN metadata size.")
        out.append((kind, blob[pos+6:end]))
        pos = end
    return out


def _tescan_strip_height(blob):
    values = []
    for kind, payload in _tescan_blocks(blob):
        if kind == 2:
            values.extend(re.findall(rb'(?m)^ImageStripSize=(\d+)\r?$', payload))
    if len(values) != 1:
        raise ValueError("Expected one TESCAN ImageStripSize field.")
    return int(values[0])


def _updated_tescan_metadata(blob):
    records = []
    for kind, payload in _tescan_blocks(blob):
        if kind == 2:
            payload, count = re.subn(rb'(?m)^ImageStripSize=\d+(\r?)$',
                                     rb'ImageStripSize=0\1', payload)
            if count != 1:
                raise ValueError("Cannot update TESCAN ImageStripSize safely.")
        records.append(struct.pack('<IH', len(payload) + 2, kind) + payload)
    return b''.join(records)


def _json_value(value):
    if isinstance(value, bytes):
        return dict(encoding='base64', value=base64.b64encode(value).decode('ascii'))
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, TiffImagePlugin.IFDRational):
        return dict(numerator=value.numerator, denominator=value.denominator)
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_without_information_bar(source_path, output_path, *, crop_row=None,
                                 tag_updates=None, processing=None, calibration=None):
    """Write a new lossless TIFF and JSON audit record; refuse overwrites.

    Without crop_row, retain the legacy TESCAN metadata-confirmed path. With
    crop_row, the caller supplies an approved boundary independently of vendor.
    Metadata adapters supply tag_updates; storage tags are always regenerated.
    Supports classic single-page little-endian uint8/uint16 grayscale TIFF.
    """
    source, output = Path(source_path), Path(output_path)
    report_path = output.with_suffix('.json')
    if source.resolve() == output.resolve():
        raise ValueError("Output must be a different file from the original.")
    if output.suffix.lower() not in ('.tif', '.tiff'):
        raise ValueError("Output must have a .tif or .tiff extension.")
    if output.exists() or report_path.exists():
        raise FileExistsError("Output TIFF or audit record already exists.")
    digest = _sha256(source)
    with open(source, 'rb') as f:
        if f.read(4) != b'II*\x00':
            raise ValueError("Exporter supports classic little-endian TIFF only.")
    with Image.open(source) as original:
        if original.format != 'TIFF' or original.n_frames != 1:
            raise ValueError("Exporter requires a single-page TIFF.")
        if original.mode not in ('L', 'I;16'):
            raise ValueError("Exporter requires uint8 or uint16 grayscale TIFF.")
        if original.tag_v2.get(262, 1) != 1:
            raise ValueError("Exporter requires standard black-is-zero grayscale.")
        if original.tag_v2.get(274, 1) != 1:
            raise ValueError("Unsupported orientation; do not silently reorient.")
        compression = int(original.tag_v2.get(259, 1))
        codecs = {1:'raw', 5:'tiff_lzw', 8:'tiff_adobe_deflate',
                  32946:'tiff_deflate', 32773:'packbits'}
        if compression not in codecs:
            raise ValueError("Unsupported or potentially lossy TIFF compression.")
        tags = dict(original.tag_v2)
        types = dict(original.tag_v2.tagtype)
        if set(tags) & {330, 34665, 34853, 40965, 513, 514}:
            raise ValueError("Linked TIFF metadata requires a dedicated exporter.")
        blob = tags.get(50431)
        image = np.asarray(original).copy()

    updates = dict(tag_updates or {})
    if crop_row is None:
        if not isinstance(blob, bytes):
            raise ValueError('Automatic legacy export requires TESCAN metadata; use the batch workflow for other vendors.')
        cropped = remove_information_bar(image)
        crop_row = cropped.shape[0]
        if _tescan_strip_height(blob) != image.shape[0] - crop_row:
            raise ValueError("Visual detection disagrees with TESCAN ImageStripSize.")
        updates[50431] = _updated_tescan_metadata(blob)
    else:
        if isinstance(crop_row, bool) or not isinstance(crop_row, (int, np.integer)) or not 0 < crop_row <= image.shape[0]:
            raise ValueError('crop_row must be an integer between 1 and the image height.')
        cropped = image[:crop_row, :].copy()
    strip_height = image.shape[0] - crop_row

    # Regenerate storage/layout tags; retain descriptive/calibration values.
    structural = {256,257,258,259,262,273,277,278,279,284,317,322,323,324,325,338,339}
    if updates.keys() & structural or not updates.keys() <= tags.keys():
        raise ValueError('Metadata updates must refer to existing non-storage tags.')
    info = TiffImagePlugin.ImageFileDirectory_v2()
    for code, value in tags.items():
        if code not in structural:
            info[code] = updates.get(code, value)
            info.tagtype[code] = types[code]
    info[262] = tags.get(262, 1)
    info[277] = tags.get(277, 1)
    if 278 in tags:
        info[278] = min(tags[278], cropped.shape[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    created_output = False
    try:
        # Exclusive open avoids accidentally overwriting an existing result.
        with open(output, 'xb') as f:
            created_output = True
            Image.fromarray(cropped).save(f, format='TIFF', compression=codecs[compression], tiffinfo=info)
        with Image.open(output) as check:
            check_tags = dict(check.tag_v2)
            reloaded = np.asarray(check)
            if reloaded.dtype != image.dtype or not np.array_equal(reloaded, image[:crop_row]):
                raise RuntimeError("Saved pixels or dtype differ from the source region.")
            for code in (258,259,262,274,277,284):
                if code in tags and check_tags.get(code) != tags[code]:
                    raise RuntimeError(f"TIFF format tag {code} changed unexpectedly.")
            for code in tags.keys() - structural:
                expected = updates.get(code, tags[code])
                if check_tags.get(code) != expected:
                    raise RuntimeError(f"Metadata tag {code} was not preserved.")
        if _sha256(source) != digest:
            raise RuntimeError("Source file changed during processing.")
    except Exception:
        # Only delete a file if this call created it.
        if created_output and output.exists():
            output.unlink()
        raise

    legacy_calibration = calibration is None
    calibration = dict(calibration or {})
    if isinstance(blob, bytes) and legacy_calibration:
        for key in ('PixelSizeX','PixelSizeY'):
            match = re.search(rb'(?m)^' + key.encode() + rb'=([^\r\n\x00]+)', blob)
            if match:
                calibration[key + '_metres'] = float(match[1])
    report = dict(version=VERSION, source_file=source.name, source_sha256=digest,
                  libraries=dict(numpy=np.__version__, Pillow=PIL.__version__),
                  output_file=output.name, output_sha256=_sha256(output),
                  original_shape=list(image.shape), cropped_shape=list(cropped.shape),
                  dtype=str(image.dtype), crop_row=crop_row, removed_rows=strip_height,
                  calibration=calibration, detector_parameters=DETECTOR_PARAMETERS,
                  verification=dict(retained_pixels_identical=True, dtype_preserved=True,
                      compression_preserved=True, source_file_unchanged=True,
                      descriptive_metadata_preserved=True,
                      updated_metadata_tag_codes=sorted(updates)),
                  processing=dict(processing or {}),
                  metadata_note='Original acquisition thumbnail retained in vendor metadata; it may still show the original footer.',
                  original_tiff_tags={str(k):dict(type=types[k], value=_json_value(v)) for k,v in tags.items()})
    with open(report_path, 'x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = save_without_information_bar(args.source, args.output)
    print(json.dumps({k:report[k] for k in ('source_file','output_file','original_shape','cropped_shape','dtype','calibration')}, indent=2))
