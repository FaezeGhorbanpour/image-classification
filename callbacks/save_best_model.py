import math
import operator
import os
from tensorflow import keras
from keras.callbacks import Callback


class SaveBestModel(Callback):
    def __init__(self, mode=max, metric='val_accuracy', wanted_metric_value=None, path='', model_name=''):
        super(SaveBestModel).__init__()
        self.mode = mode
        self.metric = metric
        self.wanted_metric_value = wanted_metric_value
        self.active = True
        self.path = path
        self.model_name = model_name

    def on_train_begin(self, logs=None):
        if self.mode == 'max':
            self.operation = operator.ge
            self.goal = 0
        else:
            self.operation = operator.le
            self.goal = math.inf


    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs.get(self.metric, None)
        if not metric_value:
            self.active = False
            print(
                'Invalid metric in "save best model" callback, You must enter this metric in the metrics of model.fit')
        else:
            if self.operation(metric_value, self.goal):
                self.goal = metric_value
                self.model.save(os.path.join(self.path, 'temporary_model.h5'))

    def on_train_end(self, logs=None):
        if self.active:
            model = keras.models.load_model(os.path.join(self.path, 'temporary_model.h5'))
            if not self.wanted_metric_value or (
                    self.wanted_metric_value and self.operation(self.goal, self.wanted_metric_value)):
                model.save(os.path.join(self.path, self.model_name + '_' + str(round(self.goal, 2)) + '.h5'))
                self.model = model
                logs[self.metric] = self.goal