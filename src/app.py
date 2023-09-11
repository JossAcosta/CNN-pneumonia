# /usr/bin/env python3

import sys

from sklearn.metrics import confusion_matrix

from .trainer import Trainer
from .model import PredictorModel
from .sample import Sample
from .plot import ConfusionMatrixPlotter


def train_model():
    training_sample = Sample(name="train")
    eval_sample = Sample(name="val")

    model = PredictorModel()

    eval_sample.load()
    loss, accuracy = model.evaluate(eval_sample)

    trainer = Trainer(sample=training_sample)
    trainer.train(model)

    model.save("model.pkl")

    return 0


def load_trained_model():
    model = PredictorModel()
    model.load("model.pkl")

    return model


def is_sick(model, image_path):
    prediction = model.predict_image_by_path(image_path)[0][0]

    if prediction == 1:
        print(
            f"La persona en la imagen {image_path} esta enferma de neumonia")
    else:
        print(f"La persona en la imagen {image_path} esta sana")


def diagnose():
    SICK_PATH = "chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg"
    HLTH_PATH = "chest_xray/test/NORMAL/IM-0016-0001.jpeg"

    model = load_trained_model()

    is_sick(model, SICK_PATH)
    is_sick(model, HLTH_PATH)

    return 0


if __name__ == "__main__":
    command = sys.argv[1]

    if command == "diagnose":
        sys.exit(diagnose())

    if command == "train":
        sys.exit(train_model())
