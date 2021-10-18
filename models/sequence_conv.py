from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, Flatten, GlobalMaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

import numpy as np
from sklearn import metrics

from models.deepmodel import DeepModel


class SequenceConv(DeepModel):
    def __init__(self, embedding):
        super().__init__(embedding)
        self.model_name = 'seq_conv'

    def model(self, params):
        model = Sequential()

        model.add(Convolution2D(params['conv_size'], 3, 3, padding='same', input_shape=(224, 224, 3), activation=params['activation']))
        model.add(Convolution2D(params['conv_size'], 3, 3, padding='same', activation=params['activation']))
        model.add(GlobalMaxPooling2D())

        base = params['conv_size']
        for i in range(params['depth']-1):
            base *= 2
            model.add(Convolution2D(base, 3, 3, padding='same', activation=params['activation']))
            model.add(Convolution2D(base, 3, 3, padding='same', activation=params['activation']))
            model.add(GlobalMaxPooling2D())

        model.add(Flatten())
        model.add(Dense(params['hidden_size'], activation=params['activation']))
        model.add(Dropout(params['dropout']))
        model.add(Dense(params['hidden_size'], activation=params['activation']))
        model.add(Dropout(params['dropout']))
        model.add(BatchNormalization())

        model.add(Dense(self.dataset.get_class_count(), activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=params['lr']),
                      metrics=['accuracy'])

        return model

    def objective(self, trial):
        print(trial.params)
        params = {
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            "depth": trial.suggest_categorical('filter_size', [1, 2, 3]),
            # "optimizer": trial.suggest_categorical('optimizer', [Adam, SGD, RMSprop]),
            "dropout": trial.suggest_categorical("dropout", [0.1, 0.3, 0.5]),
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
            "conv_size": trial.suggest_categorical("conv_size", [16, 32]),
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-1),
        }
        model = self.model(params)
        preds, probs = self.train(model)
        reals = np.argmax(self.validation_y, axis=1)
        accuracy = metrics.accuracy_score(reals, preds)
        return accuracy

    def train_test(self):
        params = self.load_params()
        model = self.model(params)
        preds, probs = self.train(model)
        return preds, probs
