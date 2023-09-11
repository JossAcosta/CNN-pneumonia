#!/usr/bin/env python3

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

    model.save("new-model.pkl")

    return model

def load_trained_model():
    model = PredictorModel()
    model.load("model.pkl")

    return model


def is_sick(model, image_path):
    prediction = model.predict_image_by_path(image_path)[0][0] == 1

    if prediction is True:
        print(f"La persona en la imágen {image_path} está enferma de neumonía")
    else:
        print(f"La persona en la imágen {image_path} está sana")


SICK_PATH = "chest-xray/val/pneumonia/person1946_bacteria_4874.jpeg"
HLTH_PATH = "chest-xray/test/normal/IM-0016-0001.jpeg"
model = load_trained_model()

if __name__ == "__main__":
    print(is_sick(model, SICK_PATH))
    print(is_sick(model, HLTH_PATH))
