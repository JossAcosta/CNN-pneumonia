import os
import pdb

import cv2
import numpy as np
from .store import load_image, list_directory


class Sample:
    def __init__(self, name: str = "test"):
        self.sample_name = name

    @property
    def name(self):
        return self.sample_name

    def load(self, name: str = None):
        sample_name = name or self.sample_name

        self.normal_images = self.load_normal_sample(sample_name=sample_name)
        self.normal_tags = np.zeros(len(self.normal_images))

        self.pneumo_images = self.load_pneumo_sample(sample_name=sample_name)
        self.pneumo_tags = np.ones(len(self.pneumo_images))

    def load_normal_sample(self, sample_name: str = "test"):
        return self.load_raw_images(
            f"chest_xray/{sample_name}/NORMAL"
        )

    def load_pneumo_sample(self, sample_name: str = "test"):
        return self.load_raw_images(
            f"chest_xray/{sample_name}/PNEUMONIA", training_tag=1
        )

    def load_raw_images(self, sample_path: str, training_tag: int = 0):
        sample = list_directory(sample_path)
        image_buffer = []
        for file_name in sample:
            image_path = os.path.join(sample_path, file_name)
            image = load_image(image_path)
            image_buffer.append(self.process_image(image))

        return np.array(image_buffer)

    def process_image(self, raw_image, image_size=(128, 128)):
        try:
            image_data = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
            image_data = cv2.resize(image_data, image_size)
            return image_data / 255.0
        except Exception as error:
            pdb.post_mortem()
            raise error

    def get_concatenated_images(self):
        sample = np.concatenate([self.normal_images, self.pneumo_images])
        return np.expand_dims(sample, axis=-1)

    def get_concatenated_tags(self):
        sample = np.concatenate([self.normal_tags, self.pneumo_tags])
        return np.expand_dims(sample, axis=-1)
