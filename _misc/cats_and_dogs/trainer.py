import matplotlib.pyplot as plt
import tensorflow as tf

import tensorflow_datasets as tfds


class CatDogTrainer:

    def __init__(self,
                 initial_epochs=10,
                 model_type='resnet50'):

        self.SPLIT_WEIGHTS = (8, 1, 1)
        self.IMG_SIZE = 224
        self.BATCH_SIZE = 32
        self.SHUFFLE_BUFFER_SIZE = 1000
        self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)

        self.model_type = model_type

        self.train_batches = None
        self.validation_batches = None
        self.test_batches = None
        self.build_data()

        self.base_learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.RMSprop(lr=self.base_learning_rate)
        self.loss = tf.keras.losses.BinaryCrossentropy()  # 'binary_crossentropy'
        self.metrics = [tf.keras.metrics.BinaryAccuracy()]  #['accuracy']
        self.initial_epochs = initial_epochs

        self.base_model = None
        self.global_average_layer = None
        self.prediction_layer = None
        self.model = None
        self.build_model()

        self.history = None

    def build_data(self):
        splits = tfds.Split.TRAIN.subsplit(weighted=self.SPLIT_WEIGHTS)

        (raw_train, raw_validation, raw_test), metadata = tfds.load('cats_vs_dogs',
                                                                    split=list(splits),
                                                                    with_info=True,
                                                                    as_supervised=True)

        def format_example(image, label):
            image = tf.cast(image, tf.float32)
            image /= 255.0  # (image / 127.5) - 1
            image = tf.image.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
            return image, label

        train = raw_train.map(format_example)
        validation = raw_validation.map(format_example)
        test = raw_test.map(format_example)

        self.train_batches = train.shuffle(self.SHUFFLE_BUFFER_SIZE).batch(self.BATCH_SIZE)
        self.validation_batches = validation.batch(self.BATCH_SIZE)
        self.test_batches = test.batch(self.BATCH_SIZE)

    def build_model(self):

        if self.model_type == 'mobile_net':
            self.base_model = tf.keras.applications.MobileNetV2(input_shape=self.IMG_SHAPE,
                                                                include_top=False,
                                                                weights='imagenet')
        elif self.model_type == 'resnet50':
            self.base_model = tf.keras.applications.ResNet50V2(input_shape=self.IMG_SHAPE,
                                                               include_top=False,
                                                               weights='imagenet')

        self.base_model.trainable = False
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.prediction_layer = tf.keras.layers.Dense(2)

        self.model = tf.keras.Sequential([
            self.base_model,
            self.global_average_layer,
            self.prediction_layer
        ])

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

    def train(self):
        self.history = self.model.fit(self.train_batches,
                                      epochs=self.initial_epochs,
                                      validation_data=self.validation_batches)

    def plot_history(self):
        acc = self.history.history['binary_accuracy']
        val_acc = self.history.history['val_binary_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0, 1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()
