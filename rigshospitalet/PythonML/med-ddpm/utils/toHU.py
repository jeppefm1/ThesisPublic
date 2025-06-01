import numpy as np
def to_hu(image):
    volume_norm = np.clip(image, -0.99999999999999, 0.99999999999999)
    volume_min = -1000
    volume_max = 2000
    
    # Invert the normalization
    return ((volume_norm + 1) / 2) * (volume_max - volume_min) + volume_min
    
     