import numpy as np
from keras.utils import np_utils


class Dataset:
    def __init__(self):
        self.data_name = ''
        self.data_path = ''
        self.output_path = ''
        self.labels_name = []
        self.train_x = None
        self.test_x = None
        self.validation_x = None
        self.train_y = None
        self.test_y = None
        self.validation_y = None

    def normalizer(self, text):
        return text

    def translator(self, text):
        return text

    def get_class_count(self):
        return len(self.labels_name)

    def get_train(self):
        return self.train_x, self.train_y

    def get_test(self):
        return self.test_x, self.test_y

    def get_validation(self):
        return self.validation_x, self.validation_y
    
    
    def labels_to_id(self):
        self.train_y = np.array([self.labels_name.index(i) for i in self.train_y])
        self.test_y = np.array([self.labels_name.index(i) for i in self.test_y])
        self.validation_y = np.array([self.labels_name.index(i) for i in self.validation_y])
        return self.train_y, self.test_y, self.validation_y

    def categorical_labels(self):
        self.labels_to_id()
        self.train_y = np_utils.to_categorical(self.train_y)
        self.test_y = np_utils.to_categorical(self.test_y)
        self.validation_y = np_utils.to_categorical(self.validation_y)
        return self.train_y, self.test_y, self.validation_y
