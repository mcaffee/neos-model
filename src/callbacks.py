import numpy as np
from keras.callbacks import Callback


class ConfusionMatrixCallback(Callback):
    def __init__(self, experiment, dataset):
        self.experiment = experiment
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.dataset)
        y_pred = np.argmax(y_pred, axis=1)

        self.experiment.log_confusion_matrix(
            self.dataset.labels,
            y_pred,
            title='Confusion Matrix, Epoch #%d' % (epoch + 1),
            file_name="confusion-matrix-%03d.json" % (epoch + 1),
        )
