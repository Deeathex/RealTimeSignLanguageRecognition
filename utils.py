import cv2
import glob
import os

import constants as constant
from tensorflow.python.keras.preprocessing import image
import numpy as np


def get_max_contour(contours, min_area=200):
    """
    Get the biggest contour
    :param contours: a vector with coordinates for the contour
    :param min_area:
    :return: the maximum possible contour
    """
    max_contour = None
    max_area = min_area
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_contour = cnt
    return max_contour


def get_letter_based_on_prediction(result):
    """
    Get the corresponding letter from alphabet based on prediction result
    :param result: a vector hot-encoded for alphabet letters
    :return: the corresponding letter
    """
    maximum = -1
    position = -1
    i = 0
    for probability in result[0]:
        if probability > maximum:
            maximum = probability
            position = i
        i += 1

    letter = chr(position + 65)
    return letter


def get_prediction(test_image, classifier):
    """
    Give a result based on the CNN model
    :param test_image: image to be predicted
    :param classifier: the model cladssifier
    :return: the corresponding result for the image
    """
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    return get_letter_based_on_prediction(result)


def save_image_from_frame(image_to_save,image_filename, image_format='.JPG', ):
    image_to_be_saved = cv2.resize(image_to_save, (50, 50))
    cv2.imwrite(constant.SAVED_IMAGES_DIRECTORY + image_filename + image_format,
                image_to_be_saved)


def flip_images_from(directory, images_format='.JPG', start_name_image=2500):
    """
    Flip all images vertical (Oy axis) from the given directory
    :param directory: directory with containing images
    :param images_format: one of JPG, PNG, JPEG, etc
    :param start_name_image: the number to start image indexing, as an example, the first would be 2500.JPG, the second
    2501.JPG, etc
    :return: -
    """
    for filename in glob.glob(directory + '/*' + images_format):
        img = cv2.imread(filename)
        vertical_img = cv2.flip(img, 1)
        cv2.imwrite(directory + '/' + str(start_name_image) + images_format,
                    vertical_img)
        start_name_image += 1


def rename_images_from(directory=constant.SAVED_IMAGES_DIRECTORY, images_format='.JPG', start_name_image=2500,
                       delete_images_after_rename=False):
    """
    Rename all images from given directory
    :param directory: directory with containing images
    :param images_format: one of JPG, PNG, JPEG, etc
    :param start_name_image: the number to start image indexing, as an example, the first would be 2500.JPG, the second
    2501.JPG, etc
    :param delete_images_after_rename: if True all the original images would be deleted after renaming, otherwise the
    images would be left there
    :return: -
    """
    image_list = []
    image_names = []

    for filename in glob.glob(directory + '*' + images_format):
        original_image = cv2.imread(filename)
        image_list.append(original_image)
        new_name = directory + str(start_name_image) + images_format
        image_names.append(new_name)
        start_name_image += 1
        if delete_images_after_rename:
            os.remove(filename)

    i = 0
    for original_image in image_list:
        cv2.imwrite(image_names[i], original_image)
        i += 1


def scale_images_from(directory, images_format='.JPG', new_size=(50, 50)):
    """
    Scale images from given directory
    :param directory: directory with containing images
    :param images_format: one of JPG, PNG, JPEG, etc
    :param new_size: a pair (x,y) where x and y represents the width and height of the new images
    :return: -
    """
    image_list = []
    image_names = []

    for filename in glob.glob(directory + '*' + images_format):
        original_image = cv2.imread(filename)
        image_list.append(original_image)
        image_names.append(filename)

    i = 0
    for original_image in image_list:
        new_image = cv2.resize(original_image, dsize=new_size)
        cv2.imwrite(image_names[i], new_image)
        i += 1
