import cv2
import numpy
import pickle
import keyboard

import gestures_recognition as gt
import constants as constant
import utils as ut


def nothing(parameter):
    pass


class HandSegmentation:
    def __init__(self):
        self.camera_output = 'Camera Output'
        self.camera_hand = 'Hand'
        self.camera_hand_train = 'Hand Train'
        self.gesture_recognition = gt.GestureRecognition('model_saved_2019-06-19.h5')
        self.hand_rectangle = None
        self.min_YCrCb = None
        self.max_YCrCb = None

    def create_windows(self):
        # Create a window to display the camera feed
        cv2.namedWindow(self.camera_output)
        cv2.namedWindow(self.camera_hand)
        cv2.namedWindow(self.camera_hand_train)

    @staticmethod
    def create_trackbars():
        # TrackBars for fixing skin color of the person
        cv2.createTrackbar('B for min', 'Camera Output', 0, 255, nothing)
        cv2.createTrackbar('G for min', 'Camera Output', 0, 255, nothing)
        cv2.createTrackbar('R for min', 'Camera Output', 0, 255, nothing)
        cv2.createTrackbar('B for max', 'Camera Output', 0, 255, nothing)
        cv2.createTrackbar('G for max', 'Camera Output', 0, 255, nothing)
        cv2.createTrackbar('R for max', 'Camera Output', 0, 255, nothing)

    @staticmethod
    def set_default_trackbars_values():
        # Default skin color values in indoor lighting
        cv2.setTrackbarPos('B for min', 'Camera Output', 0)
        cv2.setTrackbarPos('G for min', 'Camera Output', 130)
        cv2.setTrackbarPos('R for min', 'Camera Output', 103)
        cv2.setTrackbarPos('B for max', 'Camera Output', 255)
        cv2.setTrackbarPos('G for max', 'Camera Output', 182)
        cv2.setTrackbarPos('R for max', 'Camera Output', 130)

    def set_ranges_from_trackbars(self):
        # Getting min and max colors for skin
        self.min_YCrCb = numpy.array([cv2.getTrackbarPos('B for min', 'Camera Output'),
                                      cv2.getTrackbarPos('G for min', 'Camera Output'),
                                      cv2.getTrackbarPos('R for min', 'Camera Output')], numpy.uint8)
        self.max_YCrCb = numpy.array([cv2.getTrackbarPos('B for max', 'Camera Output'),
                                      cv2.getTrackbarPos('G for max', 'Camera Output'),
                                      cv2.getTrackbarPos('R for max', 'Camera Output')], numpy.uint8)

    def display_resulting_frames(self, hand_rectangle, frame, skin_region):
        # Display the resulting frame
        cv2.imshow(self.camera_hand, hand_rectangle)
        cv2.imshow(self.camera_output, frame)
        cv2.imshow(self.camera_hand_train, skin_region)

    def process_image(self):
        # Convert image to YCrCb
        image_YCrCb = cv2.cvtColor(self.hand_rectangle, cv2.COLOR_BGR2YCR_CB)
        image_YCrCb = cv2.GaussianBlur(image_YCrCb, (5, 5), 0)
        return image_YCrCb

    def save_ranges_to_file(self):
        hand_ranges = self.min_YCrCb, self.max_YCrCb
        with open(constant.RECOGNITION_SETTINGS + "hand_ranges_segmentation_settings.txt", "wb") as f:
            pickle.dump(hand_ranges, f)

    def run(self):
        self.create_windows()
        self.create_trackbars()
        self.set_default_trackbars_values()
        video_capture = cv2.VideoCapture(0)

        while video_capture.isOpened():
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            self.hand_rectangle = self.gesture_recognition.get_hand_rectangle(frame)

            image_YCrCb = self.process_image()

            self.set_ranges_from_trackbars()

            # Find region with skin tone in YCrCb image
            skin_region = cv2.inRange(image_YCrCb, self.min_YCrCb, self.max_YCrCb)
            self.display_resulting_frames(self.hand_rectangle, frame, skin_region)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if keyboard.is_pressed('s'):
                self.save_ranges_to_file()

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()


hand_segmentation = HandSegmentation()
hand_segmentation.run()
