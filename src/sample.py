import os

import numpy as np
from .store import load_image, process_image, list_directory


class Sample:
    def __init__(self, name: str = "test"):
        self.sample_name = name

    @property
    def name(self):
        return self.sample_name

    def load(self, name: str = None):
        sample_name = name or self.sample_name

        imgs, tags = self.load_normal_sample(sample_name=sample_name)
        self.normal_images = imgs
        self.normal_tags = tags

        imgs, tags = self.load_pneumo_sample(sample_name=sample_name)
        self.pneumo_images = imgs
        self.pneumo_tags = tags

    def load_normal_sample(self, sample_name: str = "test"):
        images, tags = self.load_from_raw_images(
            f"chest-xray/{sample_name}/normal"
        )
        return images, tags

    def load_pneumo_sample(self, sample_name: str = "test"):
        images, tags = self.load_from_raw_images(
            f"chest-xray/{sample_name}/pneumonia", training_tag=1
        )
        return images, tags

    def load_from_raw_images(self, sample_path: str, training_tag: int = 0):
        sample = list_directory(sample_path)
        sample_count = len(sample)
        tags = np.zeros(sample_count) if training_tag == 0 else np.ones(
            sample_count)
        image_buffer = []
        for file_name in sample:
            image_path = os.path.join(sample_path, file_name)
            image = load_image(image_path)
            image_buffer.append(process_image(image))

        images = np.array(image_buffer)

        return images, tags

    def get_concatenated_images(self):
        sample = np.concatenate([self.normal_images, self.pneumo_images])
        return np.expand_dims(sample, axis=-1)

    def get_concatenated_tags(self):
        sample = np.concatenate([self.normal_tags, self.pneumo_tags])
        return np.expand_dims(sample, axis=-1)
