import os

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

from .sample import Sample
from .plot import CountPlotter
from .model import PredictorModel



class Trainer:
    def __init__(self, sample: Sample):
        self.sample = sample
        self.datagen = self.create_data_generator()
        self.checkpoint = self.create_checkpoint()

    def train(self, model: PredictorModel):
        self.sample.load()
        images = self.sample.get_concatenated_images()
        tags = self.sample.get_concatenated_tags()

        sample_count = len(images)
        batch_size = 32
        self.datagen.fit(images)
        training = model.fit(
            self.datagen.flow(images, tags, batch_size=batch_size),
            steps_per_epoch=sample_count / batch_size,
            epochs=10,
            callbacks=[self.checkpoint],
            verbose=1,
        )

        model.set_training(training)

    def create_data_generator(self):
        return ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=30,
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
        )

    def create_checkpoint(self):
        return ModelCheckpoint(
            "modelo_normal_neumonia.hdf5",
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
        )

    def get_sample_count_plotter(self, sample: Sample = None):
        sample = sample or self.sample
        normal_count = len(sample.normal_images)
        pneumo_count = len(sample.pneumo_images)
        plotter = CountPlotter(
            title=f"Distribucion de clases en la muestra {sample.name}",
            xlabel="Cantidad de Muestras",
            ylabel="Clase",
        )
        plotter.set_data_class(
            name="NORMAL", color="skyblue", count=normal_count
        )
        plotter.set_data_class(
            name="PNEUMONIA", color="thistle", count=pneumo_count
        )

        return plotter

    def evaluate_sample(self, sample: Sample = None):
        sample = sample or self.sample
        loss, accuracy = self.model.evaluate(sample_images, sample_tags)
        return loss, accuracy

    def evaluate_model(self, sample_images, sample_tags):
        sample_count = len(sample_images)
        batch_size = 32
        history = self.model.fit(
            self.datagen.flow(sample_images, sample_tags, batch_size=batch_size),
            steps_per_epoch=sample_count / batch_size,
            epochs=10,
            validation_data=[self.checkpoint],
            verbose=1,
        ).history

        return history["loss"], history["accuracy"]
