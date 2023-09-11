import pickle

import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from .sample import Sample
from .constants import ASSETS_DIR


class PredictorModel:
    def __init__(self):
        self._model = self._create_new_sequential_model()
        self._history = None

    def fit(self, *args, **kwargs):
        return self._model.fit(*args, **kwargs)

    def predict_sample(self, sample: Sample):
        sample.load()
        images = sample.get_concatenated_images()
        prediction_tags = self._model.predict(images)
        binary_tags = (prediction_tags > 0.5).astype(int)

        return binary_tags

    def predict_image_by_path(self, image_path):
        raw_image = load_img(
            ASSETS_DIR / image_path,
            target_size=(128, 128),
            color_mode="grayscale"
        )
        raw_array = img_to_array(raw_image) / 255.0
        image_array = np.expand_dims(raw_array, axis=0)

        prediction_tags = self._model.predict(image_array)
        binary_tags = (prediction_tags > 0.5).astype(int)

        return binary_tags

    def save(self, path):
        with open(ASSETS_DIR / path, "wb") as model_stream:
            pickle.dump(
                {"model": self._model, "history": self._history}, model_stream)

    def load(self, path):
        with open(ASSETS_DIR / path, "rb") as model_stream:
            loaded_data = pickle.load(model_stream)
            self._model = loaded_data["model"]
            self._history = loaded_data["history"]

    def set_history(self, history):
        self._history = history

    def evaluate(self, sample: Sample):
        images = sample.get_concatenated_images()
        tags = sample.get_concatenated_tags()

        loss, accuracy = self._model.evaluate(images, tags)

        return loss, accuracy

    def _create_new_sequential_model(self):
        sequential_model = Sequential()
        sequential_model.add(
            Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 1)))
        sequential_model.add(MaxPooling2D((2, 2)))
        sequential_model.add(Conv2D(64, (3, 3), activation="relu"))
        sequential_model.add(MaxPooling2D((2, 2)))
        sequential_model.add(Conv2D(128, (3, 3), activation="relu"))
        sequential_model.add(MaxPooling2D((2, 2)))
        sequential_model.add(Flatten())
        sequential_model.add(Dense(128, activation="relu"))
        sequential_model.add(Dropout(0.5))
        sequential_model.add(Dense(1, activation="sigmoid"))
        sequential_model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        return sequential_model
