import tensorflow as tf
import tensorflow_hub as hub
import utils
import pathlib


class BaseModel:

    def __init__(self,
                 params,
                 hub_url='https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4'
                 ):
        self.params = params
        self.hub_url = hub_url

        self.model = None
        self.build_model()

    def build_model(self):
        self.model = tf.keras.Sequential([
            hub.KerasLayer(self.hub_url,
                           input_shape=self.params.input_shape,
                           trainable=False),
            tf.keras.layers.Dense(self.params.num_classes,
                                  activation=self.params.output_layer_activation)
        ])

    def get_model(self):
        return self.model


if __name__ == '__main__':
    params_path = pathlib.Path('../experiments/base_model/params.json')
    params = utils.Params(params_path)
    model = BaseModel(params).get_model()
    print(model.summary())
