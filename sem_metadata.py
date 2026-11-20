"""Metadata adapters: validated TESCAN binary records and unit-aware generic TIFF text.

Recognising a manufacturer's name does not imply support for its private binary
format. Other vendors use the generic reader until a validated adapter is added.
"""
from dataclasses import dataclass, field
import json
import math
import re
import xml.etree.ElementTree as ET
from PIL import Image, TiffTags
import sem_information_bar as core

VERSION = '2.0.0'


@dataclass
class Metadata:
    values: dict = field(default_factory=dict)
    manufacturer: str = 'Unknown'
    reader: str = 'generic_tiff'
    status: str = 'generic_partial'
    calibration_status: str = 'unavailable'
    footer_height_px: object = None
    warnings: list = field(default_factory=list)
    tags: dict = field(default_factory=dict)
    tag_types: dict = field(default_factory=dict)

    def csv_fields(self):
        return dict(self.values, manufacturer=self.manufacturer,
                    metadata_reader=self.reader, metadata_status=self.status,
                    calibration_status=self.calibration_status,
                    metadata_warnings='; '.join(self.warnings))


_READERS = []


def register_metadata_reader(name, predicate, reader):
    """Register a trusted reader(tags, Metadata)->None before calling the batch.

    Predicate receives the raw TIFF tags. Reader must populate common fields in
    documented units, raw fields with its own prefix, status and calibration_status.
    Increments to adapter code require bumping its version/name for batch policy.
    """
    if any(n == name for n, _, _ in _READERS):
        raise ValueError(f'Reader already registered: {name}')
    _READERS.insert(0, (name, predicate, reader))


def _manufacturer(tags):
    # Make/model/software are authoritative identification hints; do not infer
    # manufacturer from arbitrary words elsewhere in a specimen description.
    text = ' '.join(str(tags.get(k, '')) for k in (271,272,305)).lower()
    for name, words in [('TESCAN',('tescan',)), ('Thermo Fisher / FEI',('thermo fisher','fei ')),
                        ('ZEISS',('zeiss',)), ('JEOL',('jeol',)), ('Hitachi',('hitachi',))]:
        if any(word in text for word in words):
            return name
    return str(tags.get(271, '')).strip() or 'Unknown'


def _finite(value, factor=1):
    try:
        result = float(value) * factor
        return result if math.isfinite(result) else ''
    except (TypeError, ValueError):
        return ''


def _calibration_status(info):
    for axis in ('x','y'):
        key=f'pixel_size_{axis}_nm'
        value=info.values.get(key)
        if isinstance(value,(int,float)) and value<=0:
            info.values[key]=''
            info.warnings.append(f'Invalid non-positive {key}; source value retained in raw metadata.')
    valid = [isinstance(info.values.get(f'pixel_size_{axis}_nm'), (int,float)) and
             info.values[f'pixel_size_{axis}_nm'] > 0 for axis in ('x','y')]
    info.calibration_status = 'available' if all(valid) else ('partial' if any(valid) else 'unavailable')


def _tescan(tags, info):
    info.manufacturer = 'TESCAN'
    info.reader = 'tescan_binary_v1'
    for kind, payload in core._tescan_blocks(tags[50431]):
        if kind not in (2,3):
            continue
        section = 'acquisition' if kind == 2 else 'sem'
        try:
            text = payload.decode('utf-8')
        except UnicodeDecodeError:
            text = payload.decode('latin-1')
        for line in text.rstrip('\x00').splitlines():
            key, sep, value = line.partition('=')
            if sep:
                column = f'tescan.{section}.{key}'
                if column in info.values:
                    raise ValueError(f'Duplicate TESCAN field {column}')
                info.values[column] = value
    conversions = {
        'pixel_size_x_nm': ('tescan.acquisition.PixelSizeX', 1e9),
        'pixel_size_y_nm': ('tescan.acquisition.PixelSizeY', 1e9),
        'accelerating_voltage_kV': ('tescan.sem.AcceleratorVoltage', 1e-3),
        'working_distance_mm': ('tescan.sem.WD', 1e3),
        'dwell_time_us': ('tescan.sem.DwellTime', 1e6),
        'predicted_beam_current_pA': ('tescan.sem.PredictedBeamCurrent', 1e12),
        'magnification': ('tescan.acquisition.Magnification', 1),
    }
    for out,(key,factor) in conversions.items():
        info.values[out] = _finite(info.values.get(key),factor)
    for out,key in [('operator','FullUserName'),('acquisition_date','Date'),
                    ('acquisition_time','Time'),('model','DeviceModel')]:
        info.values[out] = info.values.get(f'tescan.acquisition.{key}','')
    info.values['detector'] = info.values.get('tescan.sem.Detector','')
    info.footer_height_px = core._tescan_strip_height(tags[50431])
    info.status = 'vendor_decoded'
    _calibration_status(info)


def _flatten(value, prefix=''):
    if isinstance(value,dict):
        for key, item in value.items():
            yield from _flatten(item,f'{prefix}.{key}' if prefix else str(key))
    elif isinstance(value,list):
        for i,item in enumerate(value):
            yield from _flatten(item,f'{prefix}[{i}]')
    else:
        yield prefix, '' if value is None else str(value)


def _parse_text(text):
    text = text.strip('\x00 \r\n\t')
    if text.startswith('{'):
        return dict(_flatten(json.loads(text)))
    if text.startswith('<'):
        if '<!DOCTYPE' in text.upper() or '<!ENTITY' in text.upper():
            raise ValueError('XML declarations with entities are not parsed.')
        root = ET.fromstring(text)
        result = {}
        for i,element in enumerate(root.iter()):
            name = element.tag.rsplit('}',1)[-1]
            for key,value in element.attrib.items():
                result[f'{name}[{i}].{key}'] = value
            if element.text and element.text.strip():
                result[f'{name}[{i}].text'] = element.text.strip()
        return result
    result, section = {}, ''
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(('#',';')):
            continue
        if line.startswith('[') and line.endswith(']'):
            section = line[1:-1] + '.'
            continue
        match = re.match(r'([^=:]+)\s*[=:]\s*(.*)$',line)
        if match:
            key = section + match[1].strip()
            if key in result:
                raise ValueError(f'Duplicate text field {key}')
            result[key] = match[2].strip()
    return result


NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
UNITS = {
    'length_nm': {'m':1e9,'mm':1e6,'um':1e3,'nm':1,'pm':1e-3},
    'length_mm': {'m':1e3,'mm':1,'um':1e-3,'nm':1e-6},
    'voltage_kv': {'v':1e-3,'kv':1},
    'time_us': {'s':1e6,'ms':1e3,'us':1,'ns':1e-3},
    'current_pa': {'a':1e12,'ma':1e9,'ua':1e6,'na':1e3,'pa':1},
}


def _unit(value):
    return str(value).strip().lower().replace('µ','u').replace('μ','u').replace('/pixel','').replace('/px','')


def _quantity(fields, aliases, family):
    """Extract only values with a unit in the key, value or companion field."""
    candidates = []
    for key,value in fields.items():
        leaf = key.rsplit('.',1)[-1]
        for alias in aliases:
            compact=re.sub(r'[\s_\[\]()\-]','',leaf).lower()
            base=re.sub(r'[\s_\[\]()\-]','',alias).lower()
            if not compact.startswith(base):
                continue
            suffix=compact[len(base):]
            if suffix and _unit(suffix) not in UNITS[family]:continue
            parsed = re.fullmatch(r'\s*('+NUMBER+r')\s*([A-Za-zµμ]+(?:/(?:pixel|px))?)?\s*',str(value))
            if not parsed:
                continue
            if suffix and parsed[2] and _unit(suffix)!=_unit(parsed[2]):
                return '', True
            unit = suffix or parsed[2]
            if not unit:
                for companion in (key+'Unit', key+'_unit', key+'_units'):
                    if companion in fields:
                        unit=fields[companion]
                        break
            factor = UNITS[family].get(_unit(unit or ''))
            if factor is not None:
                val = _finite(parsed[1],factor)
                if val != '':
                    candidates.append(val)
    if not candidates:
        return '', False
    if any(not math.isclose(candidates[0],v,rel_tol=1e-6,abs_tol=1e-12) for v in candidates[1:]):
        return '', True
    return candidates[0], False


def _generic(tags,info):
    info.reader = 'generic_tiff_text_v1'
    groups=[]
    for code,value in tags.items():
        if code in (271,272,305,306,315):
            continue
        text = None
        if isinstance(value,str):
            text=value
        elif isinstance(value,bytes) and code != 50431:
            try:
                candidate=value.decode('utf-8').rstrip('\x00')
                if candidate and sum(c.isprintable() or c in '\n\r\t' for c in candidate)/len(candidate)>.98:
                    text=candidate
            except UnicodeDecodeError:
                pass
        if not text:
            continue
        try:
            fields=_parse_text(text)
        except (ValueError,ET.ParseError) as exc:
            info.warnings.append(f'Text tag {code} could not be decoded: {exc}')
            continue
        for key,value in fields.items():
            info.values[f'generic.tag_{code}.{key}']=value
        groups.append(fields)
    # Keep namespaces separate while collecting unit-verified candidates.
    all_candidates={}
    conversions={
        'pixel_size_x_nm':(('PixelSizeX','PixelWidth','PhysicalSizeX'),'length_nm'),
        'pixel_size_y_nm':(('PixelSizeY','PixelHeight','PhysicalSizeY'),'length_nm'),
        'working_distance_mm':(('WorkingDistance','WD'),'length_mm'),
        'accelerating_voltage_kV':(('AcceleratingVoltage','AcceleratorVoltage','HV'),'voltage_kv'),
        'dwell_time_us':(('DwellTime',),'time_us'),
        'predicted_beam_current_pA':(('PredictedBeamCurrent',),'current_pa'),
    }
    conflicts=[]
    for name,(aliases,family) in conversions.items():
        values=[]
        for fields in groups:
            value,conflict=_quantity(fields,aliases,family)
            if conflict:conflicts.append(name)
            if value!='':values.append(value)
        if values and all(math.isclose(values[0],v,rel_tol=1e-6) for v in values) and name not in conflicts:
            if name.startswith('pixel_size') and values[0]<=0:
                continue
            info.values[name]=values[0]
        elif values or name in conflicts:
            conflicts.append(name)
    footers=[]
    common={name:[] for name in ('operator','acquisition_date','acquisition_time','model','magnification')}
    for fields in groups:
        for key,value in fields.items():
            leaf=key.rsplit('.',1)[-1].lower()
            if leaf in ('footerheightpixels','informationbarheightpixels','footer_height_px') and re.fullmatch(r'\d+',value):
                footers.append(int(value))
            if leaf in ('detector','detectortype'):
                info.values.setdefault('detector',value)
            for name,aliases in {
                'operator':('operator','fullname','fullusername','username'),
                'acquisition_date':('acquisitiondate','date'),
                'acquisition_time':('acquisitiontime','time'),
                'model':('devicemodel','microscopemodel','model'),
                'magnification':('magnification',),
            }.items():
                if leaf in aliases:common[name].append(value)
    for name,values in common.items():
        if len(set(values))==1 and not info.values.get(name):
            info.values[name]=_finite(values[0].rstrip('x× ')) if name=='magnification' else values[0]
    if footers and len(set(footers))==1:info.footer_height_px=footers[0]
    elif footers:info.warnings.append('Conflicting footer heights; manual review required.')
    _calibration_status(info)
    if any(n.startswith('pixel_size') for n in conflicts):
        info.calibration_status='conflicting'
    if conflicts:info.warnings.append('Conflicting explicit values: '+', '.join(sorted(set(conflicts))))
    info.status='generic_partial'
    info.warnings.append('Private vendor binary fields may be undecoded; full original tags are archived.')


def read_metadata(path):
    with Image.open(path) as image:
        if image.format!='TIFF':raise ValueError('This workflow currently supports TIFF inputs.')
        tags=dict(image.tag_v2)
        info=Metadata(tags=tags,tag_types=dict(image.tag_v2.tagtype))
        info.manufacturer=_manufacturer(tags)
        info.values.update(original_width_px=image.width,original_height_px=image.height,
                           file_format=image.format,page_count=image.n_frames,
                           model=str(tags.get(272,'')),operator=str(tags.get(315,'')))
        for code,value in tags.items():
            name=TiffTags.TAGS.get(code,f'Tag_{code}')
            if isinstance(value,bytes):info.values[f'tiff.{name}_byte_count']=len(value)
            elif code not in (273,279,324,325):
                info.values[f'tiff.{name}']=json.dumps(core._json_value(value),ensure_ascii=False) if isinstance(value,(tuple,list)) else str(value)
    reader = next(((name,fn) for name,predicate,fn in _READERS if predicate(tags)),None)
    if reader:
        original_values=dict(info.values)
        try:reader[1](tags,info)
        except Exception as exc:
            info.values=original_values
            info.footer_height_px=None
            info.calibration_status='unavailable'
            _generic(tags,info)
            info.status='vendor_decode_failed'
            info.warnings.append(f'{reader[0]} failed: {exc}')
    else:_generic(tags,info)
    return info


def prepare_crop_tag_updates(info,crop_row):
    """Update known geometry/footer fields; refuse structured geometry we cannot rewrite."""
    updates={}
    old_height=int(info.values['original_height_px'])
    if info.reader=='tescan_binary_v1':
        updates[50431]=core._updated_tescan_metadata(info.tags[50431])
    for code,text in info.tags.items():
        if not isinstance(text,str) or code in (271,272,305,306,315):continue
        stripped=text.strip()
        if stripped.startswith('<') and crop_row!=old_height:
            raise ValueError('XML-bearing TIFF requires a format-specific geometry exporter; original preserved for review.')
        if stripped.startswith('{'):
            value=json.loads(stripped)
            if isinstance(value,dict):
                for key in ('shape','ImageHeight','image_height','ImageLength','FooterHeightPixels','InformationBarHeightPixels','footer_height_px'):
                    if key not in value:continue
                    if key=='shape':
                        if value[key]!=[old_height,int(info.values['original_width_px'])]:
                            raise ValueError('Unsupported JSON image shape; review required.')
                        value[key]=[crop_row,int(info.values['original_width_px'])]
                    elif 'footer' in key.lower() or key=='InformationBarHeightPixels':value[key]=0
                    elif value[key]==old_height:value[key]=crop_row
                updates[code]=json.dumps(value,ensure_ascii=True)
        else:
            changed=re.sub(r'(?m)^(\s*(?:FooterHeightPixels|InformationBarHeightPixels|footer_height_px)\s*[=:]\s*)\d+',r'\g<1>0',text)
            if changed!=text:updates[code]=changed
    return updates


register_metadata_reader('tescan_binary_v1',lambda tags:isinstance(tags.get(50431),bytes),_tescan)
