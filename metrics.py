import pickle
import matplotlib.pyplot as plot

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.vis_utils import plot_model

import constants as constant


class Metrics:
    def __init__(self, model_filename, history_filename):
        self.history_filename = history_filename
        self.classifier = load_model(constant.MODEL_METRICS_DIRECTORY + model_filename)
        self.history = pickle.load(open(constant.MODEL_METRICS_DIRECTORY + history_filename, "rb"))

    def plot_model(self, plot_model_to_file):
        plot_model(self.classifier, to_file=constant.MODEL_METRICS_DIRECTORY + plot_model_to_file,
                   show_shapes=True,
                   show_layer_names=True)

    def summarize_history_for_accuracy(self, plot_filename=None):
        # list all data in history
        print(self.history.keys())
        # https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
        # summarize history for accuracy
        plot.plot(self.history['accuracy'])
        plot.plot(self.history['val_accuracy'])
        plot.title('Model accuracy')
        plot.ylabel('Accuracy')
        plot.xlabel('Epoch')
        plot.legend(['Train', 'Test'], loc='upper left')
        if plot_filename is not None:
            plot.savefig(constant.MODEL_METRICS_DIRECTORY + plot_filename, bbox_inches='tight')
        plot.show()

    def summarize_history_for_loss(self, plot_filename=None):
        # summarize history for loss
        plot.plot(self.history['loss'])
        plot.plot(self.history['val_loss'])
        plot.title('Model loss')
        plot.ylabel('Loss')
        plot.xlabel('Epoch')
        plot.legend(['Train', 'Test'], loc='upper left')
        if plot_filename is not None:
            plot.savefig(constant.MODEL_METRICS_DIRECTORY + plot_filename, bbox_inches='tight')
        plot.show()

    def print_accuracy(self):
        print(self.history['accuracy'])
        print(self.history['val_accuracy'])

    def print_loss(self):
        print(self.history['loss'])
        print(self.history['val_loss'])


metrics = Metrics('model_saved_2019-06-19.h5', 'train_history_dict_2019-06-19.txt')
# metrics.plot_model('plot_model.png')
# metrics.summarize_history_for_accuracy('model_accuracy_plot.png')
# metrics.summarize_history_for_loss('model_loss_plot.png')
metrics.print_accuracy()
metrics.print_loss()
