from datasets.dataset import Dataset
import pandas as pd
import numpy as np
import os
from keras.preprocessing import image


class TwitterLoader(Dataset):
    def __init__(self):
        super().__init__()
        self.data_name = 'twitter'

        self.data_path = '../../../../../media/external_3TB/3TB/ghorbanpoor/twitter'
        # self.data_path = '/home/faeze/PycharmProjects/fake_news_detection/data/twitter'
        self.output_path = 'datasets/twitter/'
        self.labels_name = ['real', 'fake']



        train = pd.read_csv(os.path.join(self.data_path, "twitter_train_translated.csv"))
        train_image = os.path.join(self.data_path, "images_train/")
        self.train_y = train['label'].values
        self.train_x = np.array([self.read_image(train_image, x) for x in train['image']])

        test = pd.read_csv(os.path.join(self.data_path, "twitter_test_translated.csv"))
        test_image = os.path.join(self.data_path, "images_test/")
        self.test_y = test['label'].values
        self.test_x = np.array([self.read_image(test_image, x) for x in test['image']])

        validation = pd.read_csv(os.path.join(self.data_path, "twitter_test_translated.csv"))
        validation_image = os.path.join(self.data_path, "images_test/")
        self.validation_y = validation['label'].values
        self.validation_x = np.array([self.read_image(validation_image, x) for x in validation['image']])


    def read_image(self, path, id):
        img = image.load_img(f"{path}/{id}", target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.array(x)
        return x


if __name__ == '__main__':
    data_loader = TwitterLoader()





