import numpy as np
def to_hu(image):
    rescaleSlope = 1
    rescaleIntersept = (-1024)
    maxValue = 4095

    volume_norm = np.clip(image, -0.99999999999999, 0.99999999999999)
    volume = ((volume_norm + 1) / 2) * (maxValue - 0) + 0
    hu_image = volume * rescaleSlope + rescaleIntersept
    return hu_image