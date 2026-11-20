"""Turn a leaf image into a short list of numbers (features).

A machine-learning model cannot "look" at a picture the way we do. It needs
numbers. Here we describe each image by its colours, because diseased leaves
usually look different in colour (brown spots, yellow patches).
"""
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = (64, 64)  # every image is shrunk to this size first
BINS = 16            # how many colour groups we count per channel


def extract_features(image):
    """`image` is a file path or a PIL image. Returns a 1-D numpy array."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    image = image.convert("RGB").resize(IMG_SIZE)
    pixels = np.array(image, dtype="float32")  # shape: (64, 64, 3)

    features = []

    # For each colour channel (red, green, blue):
    for channel in range(3):
        values = pixels[:, :, channel].ravel()

        # 1) Colour histogram: how many pixels fall in each brightness group.
        counts, _ = np.histogram(values, bins=BINS, range=(0, 255))
        features.extend(counts / counts.sum())  # turn counts into fractions

        # 2) Average and spread of the channel.
        features.append(values.mean() / 255)
        features.append(values.std() / 255)

    # 3) What share of pixels are "mostly green"? Healthy leaves score high.
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    features.append(np.mean((green > red) & (green > blue)))

    return np.array(features)
