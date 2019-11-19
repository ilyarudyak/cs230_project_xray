import tensorflow as tf
import tensorflow_addons as tfa
import pathlib
import utils
import numpy as np

# from model.base_model import BaseNet
from model.base_model_fine_tune import BaseNetTuned
from dataset import ChestXrayDataset

from sklearn.metrics import confusion_matrix

import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)


class Trainer:

    def __init__(self,
                 experiment_dir=pathlib.Path('experiments/fine_model'),
                 params=None,
                 ):

        # parameters
        if params:
            self.params = params
        else:
            self.params = utils.Params(str(experiment_dir / 'params.json'))

        # model
        # self.net = BaseNet(params=self.params)
        self.net = BaseNetTuned(params=self.params)
        self.model = self.net.get_model()

        # directories
        self.experiment_dir = experiment_dir
        self.weight_file = self.experiment_dir / 'weights_val_loss'
        self.history_file = self.experiment_dir / 'history.pickle'
        self.history_fine_file = self.experiment_dir / 'history_fine.pickle'

        # dataset
        if self.params.small_model:
            self.dataset = ChestXrayDataset(params=self.params,
                                            data_dir=pathlib.Path.home() / 'data/chest_xray/small_10')
        else:
            self.dataset = ChestXrayDataset(params=self.params)
        self.train_ds, self.val_ds, self.test_ds = self.dataset.build_datasets()

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=self.params.learning_rate)
        self.optimizer_fine = tf.keras.optimizers.Adam(lr=self.params.fine_learning_rate)

        # metrics and loss
        self.metrics = [
            tf.keras.metrics.CategoricalAccuracy()
                        ]
        self.loss = self.params.loss

        # callbacks
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(str(self.weight_file),
                                               save_weights_only=True,
                                               monitor='val_loss',
                                               save_best_only=True,
                                               verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.75,
                                                 patience=5,
                                                 min_lr=1e-5,
                                                 verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=1e-3,
                                             patience=15,
                                             mode='min',
                                             verbose=1)
        ]

        # history
        self.history = None
        self.history_fine = None

    def train(self, load_weights=False):

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

        if load_weights:
            self.model.load_weights(self.weight_file)

        self.history = self.model.fit(
            x=self.train_ds,
            steps_per_epoch=self.params.num_train_files // self.params.batch_size,
            epochs=self.params.epochs,
            validation_data=self.val_ds,
            validation_steps=self.params.num_val_files // self.params.batch_size,
            callbacks=self.callbacks
        )
        utils.save_history_dict(self.history, self)
        return self.history

    def train_fine(self, load_weights=False):

        self.model.compile(optimizer=self.optimizer_fine,
                           loss=self.loss,
                           metrics=self.metrics)
        if load_weights:
            self.model.load_weights(self.weight_file)

        self.history_fine = self.model.fit(
            x=self.train_ds,
            steps_per_epoch=self.params.num_train_files // self.params.batch_size,
            epochs=self.params.epochs+self.params.fine_tune_epochs,
            initial_epoch=self.history.epoch[-1]+1,
            validation_data=self.val_ds,
            validation_steps=self.params.num_val_files // self.params.batch_size,
            callbacks=self.callbacks
        )
        utils.save_history_dict(self.history_fine, self, fine=True)
        return self.history_fine

    def predict(self):
        self.model.load_weights(str(self.weight_file))
        y_pred_categorical = self.model.predict(self.test_ds,
                                                steps=self.params.num_test_files // self.params.batch_size)
        y_pred = np.argmax(y_pred_categorical, axis=1)
        return y_pred

    def get_true_test_labels(self):

        def from_categorical(y_one_hot):
            if y_one_hot[0]:
                return 0
            else:
                return 1

        y_true = np.zeros(self.params.num_test_files, dtype=np.int)
        idx = 0
        for _, label in self.test_ds.unbatch().take(self.params.num_test_files):
            y_true[idx] = from_categorical(label.numpy())
            idx += 1
        return y_true

    def get_confusion_matrix(self):
        y_true = self.get_true_test_labels()
        y_pred = self.predict()
        cm = confusion_matrix(y_true, y_pred)
        return cm

    def plot_history(self, fine=False):
        utils.plot_history(self, fine=fine)

    def unfreeze(self, fine_tune_at=100):
        self.net.unfreeze(fine_tune_at=fine_tune_at)


if __name__ == '__main__':
    trainer = Trainer(experiment_dir=pathlib.Path('experiments/small_model'))
    trainer.train()
    preds = trainer.predict()
