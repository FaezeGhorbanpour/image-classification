import json
import os
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.structs import TrialState
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from tensorflow import keras

from callbacks.print_results import PrintResults
from callbacks.save_best_model import SaveBestModel


class DeepModel:
    def __init__(self, dataset, epochs=50, batch_size=128):
        self.model_name = ''
        self.dataset = dataset
        self.dataset.categorical_labels()

        self.train_x, self.train_y = dataset.get_train()
        self.test_x, self.test_y = dataset.get_test()
        self.validation_x, self.validation_y = dataset.get_validation()

        self.epochs = epochs
        self.batch_size = batch_size

    def get_callbacks(self):
        # filepath = os.path.join(self.embedding.local_datasets.output_path,
        #                         self.get_name() + '_model_{epoch:02d}_{val_accuracy:02f}.h5')
        # checkpoint = ModelCheckpoint(filepath, verbose=1, monitor='val_loss', save_best_only=True, mode='max')
        print_results = PrintResults(self.validation_x, self.validation_y)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=self.epochs // 4, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.epochs // 2, verbose=1)
        save_best_model = SaveBestModel(mode='max', metric='val_accuracy', wanted_metric_value=0.6,
                                        path=self.dataset.output_path+'models', model_name=self.get_name())
        return [save_best_model, print_results, reduce_lr, early_stopping]

    def train(self, model):
        model.fit(self.train_x, self.train_y, validation_data=(self.test_x, self.test_y),
                  epochs=self.epochs, batch_size=self.batch_size, verbose=2,
                  callbacks=self.get_callbacks())
        try:
            model = keras.models.load_model(
                os.path.join(self.dataset.output_path, 'models/temporary_model.h5'))
        except:
            pass
        probs = model.predict(self.test_x)
        preds = np.argmax(probs, axis=1)
        return preds, probs

    def save_params(self, params):
        with open(os.path.join(self.dataset.output_path, 'params/' + self.get_name() + '.json'), 'w') as f:
            json.dump(params, f)

    def load_params(self):
        if not os.path.isfile(os.path.join(self.dataset.output_path, 'params/' + self.get_name() + '.json')):
            raise Exception('First run optuna main to find the best parameters')
        with open(os.path.join(self.dataset.output_path, 'params/' + self.get_name() + '.json'), 'r') as f:
            params = json.load(f)
        return params

    def get_name(self):
        return self.dataset.data_name + '_' + self.model_name

    def objective(self, trial):
        print('Objective function does not implemented.')
        return 0

    def optuna_main(self, n_trials=100):
        study = optuna.create_study(study_name=self.get_name(),
                                    sampler=TPESampler(),
                                    load_if_exists=True,
                                    direction="maximize",
                                    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=10)
                                    )
        study.optimize(self.objective, n_trials=n_trials)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial
        print(' Number: ', trial.number)
        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        self.save_params(trial.params)

    def train_test(self):
        print('train_test function does not implemented.')
        return [], []

    def main(self):
        preds, probs = self.train_test()

        self.train_y, self.test_y, self.validation_y = self.dataset.labels_to_id()  # todo
        probs = np.max(probs, axis=1)

        f_score_micro = f1_score(self.test_y, preds, average='micro', zero_division=0)
        f_score_macro = f1_score(self.test_y, preds, average='macro', zero_division=0)
        f_score_weighted = f1_score(self.test_y, preds, average='weighted', zero_division=0)
        accuarcy = accuracy_score(self.test_y, preds)

        s = ''
        print('accuracy', accuarcy)
        s += '\naccuracy\t' + str(accuarcy)
        print('f_score_micro', f_score_micro)
        s += '\nf_score_micro\t' + str(f_score_micro)
        print('f_score_macro', f_score_macro)
        s += '\nf_score_macro\t' + str(f_score_macro)
        print('f_score_weighted', f_score_weighted)
        s += '\nf_score_weighted\t' + str(f_score_weighted)

        fpr, tpr, thresholds = roc_curve(self.test_y, probs)
        AUC = auc(fpr, tpr)
        print('AUC', AUC)
        s += '\nAUC\t' + str(AUC)

        cohen_score = cohen_kappa_score(self.test_y, preds)
        print('cohen_score', cohen_score)
        s += '\ncohen_score\t' + str(cohen_score)

        report = classification_report(self.test_y, preds, target_names=self.dataset.labels_name,
                                       zero_division=0)
        print('classification report\n')
        print(report)
        s += '\nclassification report\t' + str(report)

        cm = confusion_matrix(self.test_y, preds)
        print('confusion matrix\n')
        print(cm)
        s += '\nconfusion matrix\t' + str(cm)

        with open(os.path.join(self.dataset.output_path, 'results/' + self.get_name() + '.txt'), 'w') as f:
            f.write(s)
