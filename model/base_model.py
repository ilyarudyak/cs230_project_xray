import tensorflow as tf
import tensorflow_hub as hub
import utils
import pathlib


class BaseNet:
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

        self.model = None
        self.build_model()

    def build_model(self):
        self.model = tf.keras.Sequential([
            hub.KerasLayer(self.hub_url,
                           input_shape=self.params.input_shape,
                           trainable=False),
            tf.keras.layers.Dense(self.num_classes,
                                  activation=self.output_layer_activation)
        ])

    def get_model(self):
        return self.model


if __name__ == '__main__':
    params_path = pathlib.Path('../experiments/base_model/params.json')
    params = utils.Params(params_path)
    model = BaseNet(params).get_model()
    print(model.summary())
