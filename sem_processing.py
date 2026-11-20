"""Vendor-independent crop decisions, review copies and exclusion masks."""
from pathlib import Path
import json
import shutil
import numpy as np
import PIL
from PIL import Image,ImageDraw
import sem_information_bar as core
from sem_metadata import prepare_crop_tag_updates


def load_valid_mask(value,shape):
    """True/nonzero means include in analysis; False/zero means exclude."""
    if isinstance(value,(str,Path)):
        with Image.open(value) as im:a=np.asarray(im).copy()
    else:a=np.asarray(value)
    if a.shape!=tuple(shape):raise ValueError('Analysis mask shape must equal the original image shape.')
    if not np.isin(a,[False,True,0,1,255]).all():raise ValueError('Mask must contain only boolean or 0/1/255 values.')
    if not a.any():raise ValueError('Analysis mask excludes every pixel.')
    return a.astype(bool)


def _preview(image,path,candidate=None):
    if image.ndim==2:
        x=image.astype(float);low,high=x.min(),x.max()
        x=np.zeros_like(x) if high<=low else (x-low)/(high-low)
        preview=Image.fromarray(np.rint(x*255).astype('uint8')).convert('RGB')
    else:preview=Image.fromarray(image).convert('RGB')
    if candidate is not None:
        ImageDraw.Draw(preview).line((0,candidate,preview.width-1,candidate),fill='red',width=3)
    preview.thumbnail((1200,1200))
    preview.save(path,format='PNG')


def _archive_copy(source,output,info,processing):
    """Preserve original bytes for review or mask-only analysis."""
    output.parent.mkdir(parents=True,exist_ok=True)
    with open(source,'rb') as src,open(output,'xb') as dst:shutil.copyfileobj(src,dst)
    digest=core._sha256(source)
    if core._sha256(output)!=digest:raise RuntimeError('Original copy verification failed.')
    with Image.open(source) as im:
        shape=(im.height,im.width)
        dtype=str(np.asarray(im).dtype)
    calibration={f'PixelSize{axis.upper()}_metres':info.values[f'pixel_size_{axis}_nm']/1e9
                 for axis in ('x','y') if isinstance(info.values.get(f'pixel_size_{axis}_nm'),(int,float))}
    return dict(version=core.VERSION,source_file=Path(source).name,source_sha256=digest,
                output_file=output.name,output_sha256=core._sha256(output),
                original_shape=list(shape),cropped_shape=list(shape),crop_row=None,
                removed_rows=0,dtype=dtype,calibration=calibration,
                libraries=dict(numpy=np.__version__,Pillow=PIL.__version__),
                processing=processing,verification=dict(retained_pixels_identical=True,
                    byte_identical_copy=True,dtype_preserved=True,source_file_unchanged=True),
                original_tiff_tags={str(k):dict(type=info.tag_types[k],value=core._json_value(v)) for k,v in info.tags.items()})


def process_source(source,destination,info,*,policy_hash,manual_row=None,
                   valid_mask=None,allow_image_only_crop=False):
    """Return an audit record and its actual (images/ or review/) TIFF path."""
    source,destination=Path(source),Path(destination)
    with Image.open(source) as im:
        # Do not apply orientation/channel transforms to the original data.
        pages=im.n_frames
        mode=im.mode
        orientation=im.tag_v2.get(274,1)
        photometric=im.tag_v2.get(262,1)
        image=np.asarray(im).copy()
    height=image.shape[0]
    candidate=None
    review_reason=''
    crop_method='none'
    selected_row=None
    mask=None
    if valid_mask is not None:
        if image.ndim!=2:raise ValueError('Masks currently require a 2D image.')
        mask=load_valid_mask(valid_mask,image.shape)
        selected_row=height
        crop_method='analysis_mask'
    elif manual_row is not None:
        if isinstance(manual_row,bool) or not isinstance(manual_row,(int,np.integer)) or not 0<manual_row<=height:
            raise ValueError('Approved crop row must be an integer from 1 to the image height.')
        selected_row=int(manual_row)
        crop_method='manual_crop' if selected_row<height else 'manual_no_crop'
    else:
        try:candidate=core._detect_information_bar(image)
        except ValueError:pass
        footer=info.footer_height_px
        if footer==0:
            selected_row=height;crop_method='metadata_no_footer'
        elif footer is not None:
            if not isinstance(footer,int) or not 0<footer<height:
                review_reason='Metadata footer height is outside the image bounds.'
            elif candidate==height-footer:
                selected_row=candidate;crop_method='metadata_and_image'
            else:review_reason='Metadata footer height and visual boundary do not agree.'
        elif candidate is not None and allow_image_only_crop:
            selected_row=candidate;crop_method='image_structure_unconfirmed'
        elif candidate is not None:
            review_reason=f'Image-only candidate at zero-based row {candidate}; approval required.'
        else:review_reason='No unique supported footer boundary. May be annotation-free or contain overlays; review required.'

    updates={}
    if pages!=1:
        selected_row=None;crop_method='none';review_reason='Multi-page/stack TIFF requires a dedicated reader and exporter.'
        mask=None
    if selected_row is not None and selected_row<height:
        with open(source,'rb') as f:header=f.read(4)
        if (header!=b'II*\x00' or mode not in ('L','I;16') or orientation!=1 or photometric!=1):
            selected_row=None;crop_method='none'
            review_reason='TIFF layout/orientation/channels need a dedicated exporter; original preserved.'
    if selected_row is not None and selected_row<height:
        try:updates=prepare_crop_tag_updates(info,selected_row)
        except ValueError as exc:
            selected_row=None;crop_method='none';review_reason=str(exc)
    status='success' if selected_row is not None else 'needs_review'
    processing=dict(status=status,crop_method=crop_method,
                    review_required=(status=='needs_review' or crop_method=='image_structure_unconfirmed'),
                    review_reason=review_reason,proposed_crop_row=candidate,
                    policy_hash=policy_hash,manufacturer=info.manufacturer,
                    metadata_reader=info.reader,metadata_status=info.status,
                    calibration_status=info.calibration_status,
                    metadata_warnings=list(info.warnings))
    calibration={f'PixelSize{axis.upper()}_metres':info.values[f'pixel_size_{axis}_nm']/1e9
                 for axis in ('x','y') if isinstance(info.values.get(f'pixel_size_{axis}_nm'),(int,float))}
    actual=destination
    if status=='needs_review':
        actual=destination.parent.parent/'review'/destination.name
        report=_archive_copy(source,actual,info,processing)
        preview=actual.with_suffix('.preview.png')
        _preview(image,preview,candidate)
        report['review_preview_file']=preview.name
    elif selected_row==height:
        report=_archive_copy(source,actual,info,processing)
    else:
        report=core.save_without_information_bar(source,actual,crop_row=selected_row,
                    tag_updates=updates,processing=processing,calibration=calibration)
    if mask is not None:
        mask_path=actual.with_suffix('.analysis_mask.tif')
        values=np.where(mask,255,0).astype('uint8')
        with open(mask_path,'xb') as f:Image.fromarray(values).save(f,format='TIFF',compression='raw')
        with Image.open(mask_path) as im:
            if not np.array_equal(np.asarray(im),values):raise RuntimeError('Mask save verification failed.')
        report['analysis_mask_file']=mask_path.name
        report['analysis_mask_sha256']=core._sha256(mask_path)
        report['analysis_included_pixels']=int(mask.sum())
    report['readable_metadata']=info.csv_fields()
    audit=actual.with_suffix('.json')
    # New records are written once; the crop writer has already created its audit.
    if audit.exists():
        audit.write_text(json.dumps(report,indent=2,ensure_ascii=True),encoding='utf-8')
    else:
        with open(audit,'x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=True)
    return report,actual
