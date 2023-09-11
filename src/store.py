#!/usr/bin/env python
import os

import cv2
import numpy as np

from .constants import ASSETS_DIR


class DataFile:
    path: str
    data: object

    def __init__(self, path: str = None, data: object = None):
        self.path = path
        self.data = data

    def load(self):
        file_name = self.path if self.path.endswith(".npy") else f"{self.path}.npy"
        self.data = np.load(ASSETS_DIR / file_name)

    def save(self):
        file_name = self.path[:-4] if self.path.endswith(".npy") else self.path
        np.save(constants.ASSETS_DIR / file_name, self.data)


def list_directory(path):
    return os.listdir(ASSETS_DIR / path)


def process_image(raw_image, image_size=(128, 128)):
    image_data = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    image_data = cv2.resize(image_data, image_size)
    return image_data / 255.0


def load_image(path):
    file_name = str(ASSETS_DIR / path)
    return cv2.imread(file_name)
