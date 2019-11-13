import matplotlib.pyplot as plt
import pickle
import json


class Params:
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def show_batch(image_batch, label_batch, dataset):
    batch_size, image_size, class_names = dataset.batch_size, dataset.img_size, dataset.class_names
    image_batch_numpy = image_batch.numpy()
    label_batch_numpy = label_batch.numpy()
    plt.figure(figsize=(10,10))
    for n in range(batch_size):
        _ = plt.subplot(batch_size // 2, batch_size // 2, n+1)

        # only for grayscale images
        # plt.imshow(image_batch_numpy[n].reshape(image_size, image_size), cmap='binary')
        plt.imshow(image_batch_numpy[n])
        plt.title(class_names[label_batch_numpy[n]][0])
        plt.axis('off')


def save_history_dict(history, trainer, param_name=None):

    if param_name:
        param = trainer.params.dict[param_name]
        filename = trainer.experiment_dir / f'history_{param_name}_{param}.pickle'
    else:
        filename = trainer.experiment_dir / 'history.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(history.history, f)


def load_history_dict(trainer):
    filename = trainer.history_file
    with open(filename, "rb") as f:
        history = pickle.load(f)
    return history


def plot_history(trainer):

    history_dict = load_history_dict(trainer=trainer)

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']

    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(len(acc))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')
    plt.xticks(epochs)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xticks(epochs)