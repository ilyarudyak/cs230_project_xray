import tensorflow as tf

import numpy as np
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE


class ChestXrayDataset:

    def __init__(self,
                 params,
                 data_dir=pathlib.Path.home()/'data/chest_xray/',
                 class_names=('NORMAL', 'PNEUMONIA'),
                 shuffle_buffer_size=32,
                 cache=True):

        self.params = params

        self.class_names = np.array(class_names)
        self.shuffle_buffer_size = shuffle_buffer_size
        self.img_size = self.params.input_shape[0]

        self.cache = cache
        self.batch_size = self.params.batch_size

        self.data_dir = data_dir
        self.train_dir = self.data_dir / 'train'
        self.train_list_ds = None
        self.train_labeled_ds = None
        self.train_ds = None

        self.val_dir = self.data_dir / 'val'
        self.val_list_ds = None
        self.val_labeled_ds = None
        self.val_ds = None

    def get_label(self, file_path):

        # convert the path to a list of path components
        parts = tf.strings.split(file_path, '/')

        # The second to last is the class-directory
        return np.array(self.class_names) == parts[-2]

    def decode_img(self, img):

        # convert the compressed string to a 3D uint8 tensor
        channels = self.params.input_shape[2]
        img = tf.image.decode_jpeg(img, channels=channels)

        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)

        # resize the image to the desired size.
        return tf.image.resize(img, [self.img_size, self.img_size])

    def process_path(self, file_path):

        label = self.get_label(file_path)

        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)

        return img, label

    def build_train_labeled_ds(self):
        self.train_list_ds = tf.data.Dataset.list_files(str(self.train_dir / '*/*'))
        self.train_labeled_ds = self.train_list_ds.map(self.process_path)

    def build_val_labeled_ds(self):
        self.val_list_ds = tf.data.Dataset.list_files(str(self.val_dir / '*/*'))
        self.val_labeled_ds = self.val_list_ds.map(self.process_path)

    def prepare_for_training(self, ds):

        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if self.cache:
            if isinstance(self.cache, str):
                ds = ds.cache(self.cache)
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=self.shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(self.batch_size)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    def build_datasets(self):

        self.build_train_labeled_ds()
        self.build_val_labeled_ds()

        self.train_ds = self.prepare_for_training(self.train_labeled_ds)
        self.val_ds = self.prepare_for_training(self.val_labeled_ds)

        return self.train_ds, self.val_ds
