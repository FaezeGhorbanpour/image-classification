import numpy as np
from sklearn import metrics
import tensorflow
from models.deepmodel import DeepModel
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class VGG(DeepModel):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.model_name = 'vgg16'
        print(self.train_x.shape)
        self.train_x = np.array([preprocess_input(i) for i in self.train_x])
        print(self.train_x.shape)
        self.test_x = np.array([preprocess_input(i) for i in self.test_x])
        self.validation_x = np.array([preprocess_input(i) for i in self.validation_x])

    def model(self, params):

        vgg_base = VGG16(weights='imagenet')
        vgg_base.trainable = False

        model = Sequential()
        model.add(vgg_base)
        model.add(Dense(params['hidden_layer'], activation=params['activation'], input_dim=(None, 1000)))
        model.add(Dropout(params['dropout']))
        model.add(Dense(self.dataset.get_class_count(), activation='softmax'))

        # model = tensorflow.keras.utils.multi_gpu_model(model, gpus=2)
        model.compile(optimizer=Adam(lr=params['lr']),
                      loss=params['loss'],
                      metrics=['accuracy'])

        # initialize_vars()
        return model

    def objective(self, trial):
        tensorflow.compat.v1.reset_default_graph()
        params = {
            "loss": 'categorical_crossentropy',
            "dropout": trial.suggest_categorical("dropout", [0.1, 0.3, 0.5]),
            "hidden_layer": trial.suggest_categorical("hidden_layer", [16, 32, 64, 128, 256]),
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-1),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid']),
        }
        model = self.model(params)
        model.fit(self.train_x, self.train_y, validation_data=(self.validation_x, self.validation_y),
                  epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=2,
                  callbacks=self.callbacks_list)
        probs = model.predict(self.validation_x)
        preds = np.argmax(probs, axis=1)
        reals = np.argmax(self.validation_y, axis=1)
        accuracy = metrics.accuracy_score(reals, preds)
        return accuracy

    def train_test(self):
        params = self.load_params()
        model = self.model(params)
        model.fit(self.train_x, self.train_y, epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                  validation_data=(self.test_x, self.test_y), callbacks=self.callbacks_list, )
        probs = model.predict(self.test_x)
        preds = np.argmax(probs, axis=1)
        return preds, probs
    
def initialize_vars():
    sess = tensorflow.compat.v1.Session(config=tensorflow.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)) #command to run codeon multiple gpu
    sess.run(tensorflow.compat.v1.local_variables_initializer())
    sess.run(tensorflow.compat.v1.global_variables_initializer())
    sess.run(tensorflow.compat.v1.tables_initializer())
    tensorflow.compat.v1.keras.backend.K.set_session(sess)