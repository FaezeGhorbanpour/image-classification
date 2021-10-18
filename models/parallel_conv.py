from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Embedding, Dense, Convolution2D, concatenate, GlobalMaxPooling2D, Dropout, \
    BatchNormalization
from tensorflow.keras.optimizers import Adam

import numpy as np
from sklearn import metrics

from models.deepmodel import DeepModel


class ParallelConv(DeepModel):
    def __init__(self, embedding):
        super().__init__(embedding)
        self.model_name = 'multi_conv'

    def model(self, params):
        graph_in = Input(shape=(224, 224, 3), name='input')

        convs = []
        for filter_size in params['filter_size']:
            x = Convolution2D(params['conv_size'], filter_size, padding='same', activation=params['activation'],
                              name='conv' + str(filter_size))(graph_in)
            convs.append(x)

        graph_out = concatenate(convs, axis=1)
        graph_out = GlobalMaxPooling2D()(graph_out)
        graph = Model(graph_in, graph_out)

        model = Sequential([
            graph,
            Dense(params['hidden_size'], activation=params['activation'], name='dense1'),
            Dropout(params['dropout'], name='dropout'),
            BatchNormalization(name='normalization'),
            Dense(self.dataset.get_class_count(), activation='softmax', name='dense2')])

        model.compile(loss='categorical_crossentropy', optimizer=Adam(params['lr'], amsgrad=True), metrics=['accuracy'])

        return model

    def objective(self, trial):
        print(trial.params)
        params = {
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh',]),
            "filter_size": trial.suggest_categorical('filter_size', [[16, 32, 64], [4, 16, 32], ]),
            # "optimizer": trial.suggest_categorical('optimizer', [Adam, SGD, RMSprop]),
            "dropout": trial.suggest_categorical("dropout", [0.1, 0.3, 0.5]),
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 64, 256]),
            "conv_size": trial.suggest_categorical("conv_size", [16, 64, 256]),
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
