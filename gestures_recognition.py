import cv2
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import numpy as np
import utils as ut

import constants as constant


class GestureRecognition:
    def __init__(self, model_filename):
        self.__classifier = load_model(constant.MODEL_METRICS_DIRECTORY + model_filename)
        self.__frame = None
        self.__hand_rectangle = None
        self.__contours = None
        self.__mask = None
        self.__x, self.__y, self.__weight, self.__height = 50, 100, 200, 200

    def get_hand_rectangle(self, frame):
        self.__frame = frame
        x = self.__x
        y = self.__y
        w = self.__weight
        h = self.__height
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

        return frame[y:y + h, x:x + w]

    def get_mask_and_contour(self, hand_rectangle):
        # applying mask and using hand segmentation to find contours
        self.__hand_rectangle = hand_rectangle
        image_ycrcb = cv2.cvtColor(hand_rectangle, cv2.COLOR_BGR2YCR_CB)
        blur = cv2.GaussianBlur(image_ycrcb, (11, 11), 0)

        skin_ycrcb_min, skin_ycrcb_max = ut.load_ranges_hand_settings_from('hand_ranges_segmentation_settings.txt')

        mask = cv2.inRange(blur, skin_ycrcb_min,
                           skin_ycrcb_max)  # detecting the hand in the bounding box using skin detection

        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, structuring_element, dst=mask)
        cv2.dilate(mask, structuring_element, mask)

        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 2)
        # using contours to capture the skin filtered image of the hand
        contour = ut.get_max_contour(contours, constant.MIN_AREA_TO_BE_MOTION)

        self.__contours = contours
        self.__mask = mask

        return mask, contour

    def recognize(self):
        image_to_be_saved = cv2.resize(self.__mask, (50, 50))
        cv2.imwrite(constant.SAVED_IMAGES_DIRECTORY + 'predict.JPG',
                    image_to_be_saved)

        test_image = image.load_img(constant.SAVED_IMAGES_DIRECTORY + 'predict.JPG',
                                    color_mode="grayscale")

        current_result = ut.get_prediction(test_image, self.__classifier)

        return current_result

    @staticmethod
    def detect_motion(last_frame, current_frame, frames_count, thresh=10):
        last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Find the absolute difference between frames
        delta_frame = cv2.absdiff(last_frame, current_frame)
        motion_mean = 0
        if frames_count % 30 == 0:
            motion_mean = np.mean(delta_frame)
            print('Diff:', motion_mean)

        if motion_mean > thresh:
            return True

        return False

    def bounding_hand(self, contour):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(self.__hand_rectangle, (x, y), (x + w, y + h), (0, 255, 0), 2)
