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
        self.histogram = ut.load_hand_histogram_from("hand_histogram")
        self.x, self.y, self.w, self.h = 50, 100, 200, 200

    def get_hand_rectangle(self, frame):
        self.frame = frame
        x = self.x
        y = self.y
        w = self.w
        h = self.h
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

        return frame[y:y + h, x:x + w]

    def get_contour(self, hand_rectangle):
        self.hand_rectangle = hand_rectangle
        # aplying mask and using hand segmentation to find contours
        image_ycrcb = cv2.cvtColor(hand_rectangle, cv2.COLOR_BGR2YCR_CB)
        blur = cv2.GaussianBlur(image_ycrcb, (11, 11), 0)

        skin_ycrcb_min, skin_ycrcb_max = ut.load_ranges_hand_settings_from("hand_ranges_segmentation_settings.txt")

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
        return hand_with_contour
        # hand_with_contour_gray = cv2.cvtColor(hand_with_contour, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(hand_with_contour_gray, (5, 5), 0)
        # ret_otsu_threshold, otsu_threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.drawContours(otsu_threshold, self.contours, -1, (255, 255, 255), -1)
        # otsu_threshold = cv2.resize(otsu_threshold, (200, 200))

    def get_image_contour_and_threshold(self):
        x = self.x
        y = self.y
        w = self.w
        h = self.h
        img = cv2.flip(self.frame, 1)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([img_hsv], [0, 1], self.histogram, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(dst, -1, disc, dst)
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.merge((thresh, thresh, thresh))
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        thresh = thresh[y:y + h, x:x + w]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        return img, contours, thresh

    def get_prediction_from_contour(self, contour, thresh):
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        save_img = thresh[y1:y1 + h1, x1:x1 + w1]
        text = ""
        if w1 > h1:
            save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0,
                                          cv2.BORDER_CONSTANT,
                                          (0, 0, 0))
        elif h1 > w1:
            save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2),
                                          cv2.BORDER_CONSTANT,
                                          (0, 0, 0))
        predicted_probability, predicted_class, letter = self.keras_predict(save_img)
        if predicted_probability * 100 > 70:
            text = letter

        return text

    def keras_predict(self, image):
        processed = ut.keras_process_image(image)
        pred_probab = self.classifier.predict(processed)[0]
        pred_class = list(pred_probab).index(max(pred_probab))
        text = ut.get_prediction(image, self.classifier)
        return max(pred_probab), pred_class, text

    def function(self):
        text = ''
        img, contours, thresh = self.get_image_contour_and_threshold()
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                text = self.get_prediction_from_contour(contour, thresh)

        return text

    def recognize(self):
        image_to_be_saved = cv2.resize(self.mask, (50, 50))
        cv2.imwrite(constant.SAVED_IMAGES_DIRECTORY + 'predict.JPG',
                    image_to_be_saved)

        test_image = image.load_img(constant.SAVED_IMAGES_DIRECTORY + 'predict.JPG',
                                    color_mode="grayscale")

        current_result = ut.get_prediction(test_image, self.classifier)

        return current_result

    @staticmethod
    def detect_motion(last_frame, current_frame, frames_count):
        last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Find the absolute difference between frames
        diff = cv2.absdiff(last_frame, current_frame)
        motion_mean = 0
        if frames_count % 30 == 0:
            print('Current_frame: ', np.mean(current_frame))
            motion_mean = np.mean(diff)
            print('Diff:', motion_mean)

        if motion_mean > 10:
            return True

        return False
