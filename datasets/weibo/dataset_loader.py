from datasets.dataset import Dataset
import pandas as pd
import numpy as np
import os
from keras.preprocessing import image


class WeiboLoader(Dataset):
    def __init__(self):
        super().__init__()
        self.data_name = 'twitter'

        self.data_path = '../../../../../media/external_3TB/3TB/ghorbanpoor/weibo'
        # self.data_path = '/home/faeze/PycharmProjects/fake_news_detection/data/weibo'
        self.output_path = 'datasets/weibo/'
        self.labels_name = ['real', 'fake']



        train = pd.read_csv(os.path.join(self.data_path, "weibo_train.csv"))
        self.train_y = train['label'].values
        self.train_x = np.array([self.read_image(train, x) for x in range(train.shape[0])])

        test = pd.read_csv(os.path.join(self.data_path, "weibo_test.csv"))
        self.test_y = test['label'].values
        self.test_x = np.array([self.read_image(test, x) for x in range(test.shape[0])])

        validation = pd.read_csv(os.path.join(self.data_path, "weibo_test.csv"))
        self.validation_y = validation['label'].values
        self.validation_x = np.array([self.read_image(validation, x) for x in range(validation.shape[0])])


    def read_image(self, data, id):
        if data.label.iloc[id] == 'fake':
            path = os.path.join(self.data_path, "rumor_images")
            img = image.load_img(f"{path}/{data.image[id]}", target_size=(224, 224))
        else:
            path = os.path.join(self.data_path, "nonrumor_images")
            img = image.load_img(f"{path}/{data.image[id]}", target_size=(224, 224))

        x = image.img_to_array(img)
        x = np.array(x)
        return x


if __name__ == '__main__':
    data_loader = WeiboLoader()





