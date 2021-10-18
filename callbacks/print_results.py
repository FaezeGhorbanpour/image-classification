import os

import pandas as pd
import tensorflow
from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc, cohen_kappa_score, \
    classification_report, confusion_matrix


class PrintResults(Callback):
    def __init__(self, validation_x, validation_y, wanted_accuracy=0.60, path='', model_name='', save_results=False):
        super(PrintResults).__init__()
        self.validation_x = validation_x
        self.validation_y = validation_y
        self.accuracy = wanted_accuracy
        self.path = path
        self.mode_name = model_name
        self.save_results = save_results

    def on_epoch_end(self, epoch, logs=None):
        p = np.asarray(self.model.predict(self.validation_x))
        probs = np.max(p, axis=1)
        preds = np.argmax(p, axis=1)

        test_y = np.argmax(self.validation_y, axis=1)
        accuarcy = accuracy_score(test_y, preds)

        if accuarcy >= self.accuracy:
            f_score_micro = f1_score(test_y, preds, average='micro', zero_division=0)
            f_score_macro = f1_score(test_y, preds, average='macro', zero_division=0)
            f_score_weighted = f1_score(test_y, preds, average='weighted', zero_division=0)

            s = ''
            print('accuracy', accuarcy)
            s += '\naccuracy\t' + str(accuarcy)
            print('f_score_micro', f_score_micro)
            s += '\nf_score_micro\t' + str(f_score_micro)
            print('f_score_macro', f_score_macro)
            s += '\nf_score_macro\t' + str(f_score_macro)
            print('f_score_weighted', f_score_weighted)
            s += '\nf_score_weighted\t' + str(f_score_weighted)

            fpr, tpr, thresholds = roc_curve(test_y, probs)
            AUC = auc(fpr, tpr)
            print('AUC', AUC)
            s += '\nAUC\t' + str(AUC)

            cohen_score = cohen_kappa_score(test_y, preds)
            print('cohen_score', cohen_score)
            s += '\ncohen_score\t' + str(cohen_score)

            report = classification_report(test_y, preds, zero_division=0)
            print('classification report\n')
            print(report)
            s += '\nclassification report\t' + str(report)

            cm = confusion_matrix(test_y, preds)
            print('confusion matrix\n')
            print(cm)
            s += '\nconfusion matrix\t' + str(cm)

            if self.save_results:
                with open(os.path.join(self.path, self.model_name + '+report_epoch_' + str(epoch) + '_accuracy_' + str(
                        round(accuarcy, 2))) + '.txt', 'w') as f:
                    f.write(s)

                pd.DataFrame(p).to_csv(os.path.join(self.path, self.model_name + '_prediction_epoch_' + str(
                    epoch) + '_accuracy_' + str(round(accuarcy, 2)) + '.csv'))
                pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(os.path.join(self.path, self.model_name + '_auc_epoch_' + str(
                    epoch) + '_accuracy_' + str(round(accuarcy, 2)) + '.csv'))

            self.accuracy = accuarcy
        return
