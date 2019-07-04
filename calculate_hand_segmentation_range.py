import cv2
import numpy
import pickle
import keyboard

import gestures_recognition as gt
import constants as constant


def nothing(parameter):
    pass


B_min = 'B min'
G_min = 'G min'
R_min = 'R min'
B_max = 'B max'
G_max = 'G max'
R_max = 'R max'


class HandSegmentation:
    def __init__(self):
        self.__camera_output = 'Camera Output'
        self.__camera_hand = 'Hand'
        self.__camera_hand_train = 'Hand Train'
        # self.__gesture_recognition = gt.GestureRecognition('model_saved_2019-06-19.h5')
        self.__gesture_recognition = gt.GestureRecognition('model_saved_2019-06-20.h5')
        self.__hand_rectangle = None
        self.__min_YCrCb = None
        self.__max_YCrCb = None

    def __create_windows(self):
        # Create a window to display the camera feed
        cv2.namedWindow(self.__camera_output)
        cv2.namedWindow(self.__camera_hand)
        cv2.namedWindow(self.__camera_hand_train)

    @staticmethod
    def __create_trackbars():
        # TrackBars for fixing skin color of the person
        cv2.createTrackbar(B_min, 'Camera Output', 0, 255, nothing)
        cv2.createTrackbar(G_min, 'Camera Output', 0, 255, nothing)
        cv2.createTrackbar(R_min, 'Camera Output', 0, 255, nothing)
        cv2.createTrackbar(B_max, 'Camera Output', 0, 255, nothing)
        cv2.createTrackbar(G_max, 'Camera Output', 0, 255, nothing)
        cv2.createTrackbar(R_max, 'Camera Output', 0, 255, nothing)

    @staticmethod
    def __set_default_trackbars_values():
        # Default skin color values in indoor lighting
        cv2.setTrackbarPos(B_min, 'Camera Output', 0)
        cv2.setTrackbarPos(G_min, 'Camera Output', 130)
        cv2.setTrackbarPos(R_min, 'Camera Output', 103)
        cv2.setTrackbarPos(B_max, 'Camera Output', 255)
        cv2.setTrackbarPos(G_max, 'Camera Output', 182)
        cv2.setTrackbarPos(R_max, 'Camera Output', 130)

    def __set_ranges_from_trackbars(self):
        # Getting min and max colors for skin
        self.__min_YCrCb = numpy.array([cv2.getTrackbarPos(B_min, 'Camera Output'),
                                        cv2.getTrackbarPos(G_min, 'Camera Output'),
                                        cv2.getTrackbarPos(R_min, 'Camera Output')], numpy.uint8)
        self.__max_YCrCb = numpy.array([cv2.getTrackbarPos(B_max, 'Camera Output'),
                                        cv2.getTrackbarPos(G_max, 'Camera Output'),
                                        cv2.getTrackbarPos(R_max, 'Camera Output')], numpy.uint8)

    def __display_resulting_frames(self, hand_rectangle, frame, skin_region):
        # Display the resulting frame
        cv2.imshow(self.__camera_hand, hand_rectangle)
        cv2.imshow(self.__camera_output, frame)
        cv2.imshow(self.__camera_hand_train, skin_region)

    def __process_image(self):
        # Convert image to YCrCb
        image_YCrCb = cv2.cvtColor(self.__hand_rectangle, cv2.COLOR_BGR2YCR_CB)
        image_YCrCb = cv2.GaussianBlur(image_YCrCb, (5, 5), 0)
        return image_YCrCb

    def __save_ranges_to_file(self):
        hand_ranges = self.__min_YCrCb, self.__max_YCrCb
        with open(constant.RECOGNITION_SETTINGS + "hand_ranges_segmentation_settings.txt", "wb") as f:
            pickle.dump(hand_ranges, f)

    def run(self):
        self.__create_windows()
        self.__create_trackbars()
        self.__set_default_trackbars_values()
        video_capture = cv2.VideoCapture(0)

        frames_count = 0
        s_was_pressed = False
        while video_capture.isOpened():
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            frames_count += 1
            if frames_count % 30 == 0:
                s_was_pressed = False
                frames_count = 0

            self.__hand_rectangle = self.__gesture_recognition.get_hand_rectangle(frame)

            image_YCrCb = self.__process_image()

            self.__set_ranges_from_trackbars()

            # Find region with skin tone in YCrCb image
            skin_region = cv2.inRange(image_YCrCb, self.__min_YCrCb, self.__max_YCrCb)
            if s_was_pressed:
                cv2.putText(frame, 'Values for hand segmentation range saved!', (0, 470), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (255, 255, 255))

            cv2.putText(frame, 'Select min and max values for hand color by placing your hand in the box', (0, 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255))
            cv2.putText(frame, 'and by setting values for trackers. When your hand is fully contoured press', (0, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255))
            cv2.putText(frame, 'S to save the range.', (0, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255))

            self.__display_resulting_frames(self.__hand_rectangle, frame, skin_region)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if keyboard.is_pressed('s'):
                s_was_pressed = True
                self.__save_ranges_to_file()

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()


hand_segmentation = HandSegmentation()
hand_segmentation.run()
