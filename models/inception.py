import numpy as np
from sklearn import metrics

from models.deepmodel import DeepModel
from tensorflow.keras.applications.inception_v3 import preprocess_input, InceptionV3
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class Inception(DeepModel):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.model_name = 'inception'
        self.train_x = np.array([preprocess_input(i) for i in self.train_x])
        self.test_x = np.array([preprocess_input(i) for i in self.test_x])
        self.validation_x = np.array([preprocess_input(i) for i in self.validation_x])

    def model(self, params):
        inception_base = InceptionV3(weights='imagenet')
        inception_base.trainable = False

        model = Sequential()
        model.add(inception_base)
        model.add(Dense(params['hidden_layer'], activation=params['activation'], input_dim=(None, 1000)))
        model.add(Dropout(params['dropout']))
        model.add(Dense(self.dataset.get_class_count(), activation='softmax'))
        model.compile(optimizer=Adam(lr=params['lr']),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def objective(self, trial):
        params = {
            "loss": 'categorical_crossentropy',
            "dropout": trial.suggest_categorical("dropout", [0.1, 0.3, 0.5]),
            "hidden_layer": trial.suggest_categorical("hidden_layer", [32, 64, 128, 256]),
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-1),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid']),
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