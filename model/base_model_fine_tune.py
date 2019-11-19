import tensorflow as tf
import tensorflow_hub as hub
import utils
import pathlib


class BaseNetTuned:
    """
    important notice: we can't use resnet50 with images that contain only one channel,
    so we read them in RGB mode - see here:
    https://stackoverflow.com/questions/51995977/how-can-i-use-a-pre-trained-neural-network-with-grayscale-images
    """

    def __init__(self,
                 params,
                 hub_url='https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4',
                 output_layer_activation='softmax',
                 num_classes=2
                 ):
        self.params = params
        self.hub_url = hub_url
        self.num_classes = num_classes
        self.output_layer_activation = output_layer_activation

        self.base_model = None
        self.model = None
        self.build_model()

    def build_model(self):
        self.base_model = tf.keras.applications.ResNet50(input_shape=self.params.input_shape,
                                                         include_top=False,
                                                         weights='imagenet')
        self.base_model.trainable = False

        self.model = tf.keras.Sequential([
            self.base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(self.num_classes,
                                  activation=self.output_layer_activation)
        ])

    def get_model(self):
        return self.model

    def unfreeze(self, fine_tune_at=100):
        self.base_model.trainable = True

        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False


if __name__ == '__main__':
    params_path = pathlib.Path('../experiments/base_model/params.json')
    params = utils.Params(params_path)
    model = BaseNetTuned(params).get_model()
    print(model.summary())
