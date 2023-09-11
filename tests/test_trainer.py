import pdb
from sklearn.metrics import confusion_matrix

from src.trainer import Trainer
from src.model import PredictorModel
from src.sample import Sample
from src.plot import ConfusionMatrixPlotter

def test_trainer_evaluate_sample():
    training_sample = Sample(name="train")
    eval_sample = Sample(name="val")
    test_sample = Sample(name="test")

    model = PredictorModel()

    eval_sample.load()
    loss, accuracy = model.evaluate(eval_sample)

    trainer = Trainer(sample=training_sample)
    trainer.train(model)

    prediction_tags = model.predict(test_sample)
    test_tags = test_sample.get_concatenated_tags()

    confusion = confusion_matrix(test_tags, prediction_tags)
    plotter = ConfusionMatrixPlotter()
    plotter.plot(confusion)
    plotter.show_plot()
