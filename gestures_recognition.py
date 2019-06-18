import cv2
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import numpy as np
import utils as ut

import constants as constant


class GestureRecognition:
    def __init__(self, model_filename):
        self.classifier = load_model(constant.MODEL_METRICS_DIRECTORY + model_filename)
        self.frame = None
        self.hand_rectangle = None
        self.contours = None
        self.mask = None

    def get_hand_rectangle(self, frame):
        self.frame = frame
        x = 50
        y = 100
        w = 200
        h = 200
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

        return frame[y:y + h, x:x + w]

    def get_contour(self, hand_rectangle):
        self.hand_rectangle = hand_rectangle
        # aplying mask and using hand segmentation to find contours
        image_ycrcb = cv2.cvtColor(hand_rectangle, cv2.COLOR_BGR2YCR_CB)
        blur = cv2.GaussianBlur(image_ycrcb, (11, 11), 0)

        skin_ycrcb_min = np.array((0, 50, 67))
        skin_ycrcb_max = np.array((255, 173, 130))
        mask = cv2.inRange(blur, skin_ycrcb_min,
                           skin_ycrcb_max)  # detecting the hand in the bounding box using skin detection

        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 2)
        contour = ut.get_max_contour(contours, 4000)  # using contours to capture the skin filtered image of the hand
        self.contours = contours
        self.mask = mask
        return mask, contour

    def otsu_and_stuff(self, contour):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(self.hand_rectangle, (x, y), (x + w, y + h), (0, 255, 0), 2)
        hand_with_contour = self.hand_rectangle[y:y + h, x:x + w]
        hand_with_contour = cv2.bitwise_and(hand_with_contour, hand_with_contour, mask=self.mask[y:y + h, x:x + w])
        hand_with_contour_gray = cv2.cvtColor(hand_with_contour, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(hand_with_contour_gray, (5, 5), 0)
        ret_otsu_threshold, otsu_threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.drawContours(otsu_threshold, self.contours, -1, (255, 255, 255), -1)
        otsu_threshold = cv2.resize(otsu_threshold, (200, 200))

    def recognize(self):
        image_to_be_saved = cv2.resize(self.mask, (50, 50))
        cv2.imwrite(constant.SAVED_IMAGES_DIRECTORY + 'predict.JPG',
                    image_to_be_saved)

        test_image = image.load_img(constant.SAVED_IMAGES_DIRECTORY + 'predict.JPG',
                                    color_mode="grayscale")

        current_result = ut.get_prediction(test_image, self.classifier)

        return current_result
