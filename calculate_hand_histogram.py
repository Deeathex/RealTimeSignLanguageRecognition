import cv2
import numpy as np
import pickle

import constants as constant


class HandHistogramController:
    def __init__(self, width_hand_area, height_hand_area):
        self.image = None
        self.width_hand_area = width_hand_area
        self.height_hand_area = height_hand_area

    def show_hand_area(self):
        x, y, w, h = 400, 120, 10, 10
        d = 10
        image_cropped = None
        crop = None
        for i in range(self.height_hand_area):
            for j in range(self.width_hand_area):
                if np.any(image_cropped is None):
                    image_cropped = self.image[y:y + h, x:x + w]
                else:
                    image_cropped = np.hstack((image_cropped, self.image[y:y + h, x:x + w]))
                x += w + d
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 0), 1)
            if np.any(crop is None):
                crop = image_cropped
            else:
                crop = np.vstack((crop, image_cropped))
            image_cropped = None
            x = 400
            y += h + d
        return crop

    def get_hand_hist(self):
        cam = cv2.VideoCapture(1)
        if not cam.read()[0]:
            cam = cv2.VideoCapture(0)
        flag_pressed_c, flag_pressed_s = False, False
        image_cropped = None
        while True:
            img = cam.read()[1]
            img = cv2.flip(img, 1)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = None

            keypress = cv2.waitKey(1)
            if keypress == ord('c'):
                hsv_crop = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2HSV)
                flag_pressed_c = True
                hist = cv2.calcHist([hsv_crop], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            elif keypress == ord('s'):
                break
            if flag_pressed_c:
                dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
                disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
                cv2.filter2D(dst, -1, disc, dst)
                blur = cv2.GaussianBlur(dst, (11, 11), 0)
                blur = cv2.medianBlur(blur, 15)
                ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh = cv2.merge((thresh, thresh, thresh))
                cv2.imshow("Thresh", thresh)
            if not flag_pressed_s:
                self.image = img
                image_cropped = self.show_hand_area()
            cv2.imshow("Set hand histogram", img)
        cam.release()
        cv2.destroyAllWindows()
        with open(constant.HISTOGRAM_DIRECTORY + "hand_histogram", "wb") as f:
            pickle.dump(hist, f)


hand_histogram_controller = HandHistogramController(5, 10)
hand_histogram_controller.get_hand_hist()
