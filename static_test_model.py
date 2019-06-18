from tensorflow.python.keras.models import load_model
import numpy as np
from tensorflow.python.keras.preprocessing import image

import constants as constant
import utils


class StaticTestModel:
    def __init__(self, model_filename):
        self.model_filename = model_filename
        self.classifier = load_model(constant.MODEL_METRICS_DIRECTORY + model_filename)

    def predict(self, image_path):
        test_image = image.load_img(image_path, grayscale=True)
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = self.classifier.predict(test_image)
        print(utils.get_letter_based_on_prediction(result))


test_model_A = StaticTestModel('model_saved_2019-06-18.h5')
