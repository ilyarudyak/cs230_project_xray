import tensorflow as tf
import pathlib
import utils

from model.base_model import BaseModel
from dataset import ChestXrayDataset


class Trainer:

    def __init__(self,
                 experiment_dir=pathlib.Path('experiments/elastic_def'),
                 params=None,
                 ):

        # parameters
        if params:
            self.params = params
        else:
            self.params = utils.Params(experiment_dir / 'params.json')

        # model
        self.model = BaseModel()

        # directories
        self.experiment_dir = experiment_dir
        self.weight_file = self.experiment_dir / 'weights'

        # dataset
        self.dataset = ChestXrayDataset()

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=self.params.learning_rate)

        # metrics and loss
        self.metrics = ['accuracy']
        self.loss = self.params.loss
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

        # callbacks
        # weight_file = str(self.experiment_dir / 'weights')
        # self.callbacks = [
        #     tf.keras.callbacks.TensorBoard(log_dir=str(self.experiment_dir),
        #                                    update_freq='epoch'),
        #     tf.keras.callbacks.ModelCheckpoint(f'{weight_file}_val_dice_coef.hdf5',
        #                                        save_weights_only=True,
        #                                        monitor='val_dice_coef',
        #                                        save_best_only=True,
        #                                        mode='max',
        #                                        verbose=1),
        #     tf.keras.callbacks.ModelCheckpoint(f'{weight_file}_val_loss.hdf5',
        #                                        save_weights_only=True,
        #                                        monitor='val_loss',
        #                                        save_best_only=True,
        #                                        verbose=1),
        #     tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
        #                                          factor=0.75,
        #                                          patience=5,
        #                                          cooldown=3,
        #                                          min_lr=1e-5,
        #                                          verbose=1),
        #     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
        #                                          factor=0.75,
        #                                          patience=5,
        #                                          cooldown=3,
        #                                          min_lr=1e-5,
        #                                          verbose=1),
        #     tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        #                                      min_delta=1e-3,
        #                                      patience=20,
        #                                      mode='min',
        #                                      verbose=1)
        # ]

    def train(self, load_weights=False):
        pass
        # if load_weights:
        #     self.model.load_weights(self.weight_file)
        #
        # train_generator = self.dataset.generator(mode='training', batch_size=self.params.batch_size)
        # valid_generator = self.dataset.generator(mode='validation', batch_size=self.params.batch_size_val)
        #
        # history = self.model.fit_generator(generator=train_generator,
        #                                    steps_per_epoch=self.params.steps_per_epoch,
        #                                    epochs=self.params.epochs,
        #                                    initial_epoch=self.params.start_epoch,
        #                                    validation_data=valid_generator,
        #                                    validation_steps=self.params.validation_steps,
        #                                    callbacks=self.callbacks)
        # return history

    def predict(self, weight_file):
        pass
        # self.model.load_weights(weight_file)
        # self.dataset.load_data_test()
        # test_masks = self.model.predict(self.dataset.image_data_test / 255.0,
        #                                 batch_size=self.params.batch_size_test)
        # test_masks = test_masks.round()
        # tiff.imsave(self.pred_file, test_masks)
