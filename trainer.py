import tensorflow as tf
import tensorflow_addons as tfa
import pathlib
import utils

from model.base_model import BaseNet
from dataset import ChestXrayDataset

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


class Trainer:

    def __init__(self,
                 experiment_dir=pathlib.Path('experiments/base_model'),
                 params=None,
                 ):

        # parameters
        if params:
            self.params = params
        else:
            self.params = utils.Params(experiment_dir / 'params.json')

        # model
        self.model = BaseNet(params=self.params).get_model()

        # directories
        self.experiment_dir = experiment_dir
        self.weight_file = self.experiment_dir / 'weights_val_loss'
        self.history_file = self.experiment_dir / 'history.pickle'

        # dataset
        if self.params.small_model:
            self.dataset = ChestXrayDataset(params=self.params,
                                            data_dir=pathlib.Path.home()/'data/chest_xray')
        else:
            self.dataset = ChestXrayDataset(params=self.params)
        self.train_ds, self.val_ds = self.dataset.build_datasets()

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=self.params.learning_rate)

        # metrics and loss
        self.metrics = [tf.keras.metrics.Accuracy(),
                        tf.keras.metrics.Precision(),
                        tf.keras.metrics.Recall(),
                        tfa.metrics.f_scores.F1Score(num_classes=2,
                                                     average=None),
                        tfa.metrics.multilabel_confusion_matrix.MultiLabelConfusionMatrix(num_classes=2)
                        ]
        self.loss = self.params.loss
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

        # callbacks
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(str(self.weight_file),
                                               save_weights_only=True,
                                               monitor='val_loss',
                                               save_best_only=True,
                                               verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.75,
                                                 patience=3,
                                                 min_lr=1e-5,
                                                 verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=1e-3,
                                             patience=5,
                                             mode='min',
                                             verbose=1)
        ]

    def train(self, load_weights=False):
        if load_weights:
            self.model.load_weights(self.weight_file)

        history = self.model.fit(x=self.train_ds,
                                 steps_per_epoch=self.params.num_train_files//self.params.batch_size,
                                 epochs=self.params.epochs,
                                 validation_data=self.val_ds,
                                 validation_steps=self.params.num_val_files//self.params.batch_size,
                                 callbacks=self.callbacks)
        utils.save_history_dict(history, self)
        return history

    def predict(self, dataset):
        self.model.load_weights(self.weight_file)

        # test_masks = self.model.predict(self.dataset.image_data_test / 255.0,
        #                                 batch_size=self.params.batch_size_test)
        # test_masks = test_masks.round()
        # tiff.imsave(self.pred_file, test_masks)

    def plot_history(self):
        utils.plot_history(self)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()

