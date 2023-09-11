#!/usr/bin/env python
import os

import cv2

from .constants import ASSETS_DIR


def list_directory(path):
    return os.listdir(ASSETS_DIR / path)


def load_image(path):
    file_name = str(ASSETS_DIR / path)
    image = cv2.imread(file_name)

    assert image is not None, file_name

    return image
