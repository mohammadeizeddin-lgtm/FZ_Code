from os import path
from pathlib import Path
import tifffile 
import numpy as np

def parse_path(path_str):
    name = Path(path_str).stem
    split = name.split('_')

    detector = split[0] # what SEM detector was used
    state = split[1] #Fresh or cycled
    chemistry = split[2] #LFP, NMC, etc.
    image_ID = int(split[3]) #Image ID

    return detector, state, chemistry, image_ID

def load_tiff(path_str):
    if not path.exists(path_str):
        raise FileNotFoundError(f"File not found: {path_str}")

    image = tifffile.imread(path_str)
    return image

def normalize_image(image, percentile_range=(0, 100)):

    print(f"Image shape: {image.shape}, Image dtype: {image.dtype}") # To check dimensions and image original bit depth
    image_float = image.astype(np.float32) # Convert to float32 for normalization

    if percentile_range != (0, 100):
        low = np.percentile(image_float, percentile_range[0])
        high = np.percentile(image_float, percentile_range[1])
        if high - low == 0:
            raise ValueError("Percentile range results in no variation; cannot normalize.")
        else:
            clipped_image = np.clip(image_float, low, high)
            normalised_image = (clipped_image - low) / (high - low)
    else:
        low, high = image_float.min(), image_float.max()
        if high - low == 0:
                raise ValueError("Image has no variation; cannot normalize.")
        else:
            normalised_image = (image_float - low) / (high - low) # Normalize to [0, 1]
    return normalised_image

def save_tiff(image, path_str):
    tifffile.imwrite(
        path_str, 
        image.astype(np.float32)) # Save as float32 to preserve normalized values
    
        
    




