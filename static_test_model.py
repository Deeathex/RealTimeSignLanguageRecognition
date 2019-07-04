from tensorflow.python.keras.models import load_model
import numpy as np
import os
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
        current_result = utils.get_letter_based_on_prediction(result)
        return utils.get_correct_character_for_special_characters(current_result)

    def test_model_with_alphabet(self, images_format='.JPG'):
        global_correct_classification_count = 0
        global_total_examples_count = 0
        for subdir, dirs, files in os.walk(constant.TEST_DATA_DIRECTORY):
            split_list = subdir.split('TestData/')
            actual_letter = split_list[1]
            print(actual_letter)
            correct_classification_count = 0
            total_examples_count = 0

            if files:
                for file in files:
                    image_path = os.path.join(subdir, file)
                    predicted_letter = self.predict(image_path)
                    total_examples_count += 1

                    if actual_letter == predicted_letter:
                        correct_classification_count += 1

                print('Correctly classified: ' + str(correct_classification_count))
                acc = correct_classification_count / total_examples_count
                global_correct_classification_count += correct_classification_count
                global_total_examples_count += total_examples_count
                print('Accuracy: ' + str(acc))

        print('Global correctly classified: ' + str(global_correct_classification_count) + "/" + str(
            global_total_examples_count))
        global_acc = global_correct_classification_count / global_total_examples_count
        print('Global accuracy: ' + str(global_acc))


# test_model_A = StaticTestModel('model_saved_2019-06-18.h5')
test_model = StaticTestModel('model_saved_2019-06-20.h5')
test_model.test_model_with_alphabet()
