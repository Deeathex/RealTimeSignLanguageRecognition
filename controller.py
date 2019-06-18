import cv2

import gestures_recognition as gest_rec
import keyboard

import utils as ut


class Controller:
    def __init__(self):
        self.gesture_recognition = gest_rec.GestureRecognition('model_saved_2019-06-18.h5')

    def run(self):
        video_capture = cv2.VideoCapture(0)
        frames = 0
        images_saved_count = 0
        old_result = ""

        while video_capture.isOpened():
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            frames += 1

            hand_rectangle = self.gesture_recognition.get_hand_rectangle(frame)

            mask, contour = self.gesture_recognition.get_contour(hand_rectangle)

            if contour is not None and contour.all is not None:
                self.gesture_recognition.otsu_and_stuff(contour)

            # Display the resulting frame
            cv2.imshow('Video', hand_rectangle)
            cv2.imshow('Mask', mask)
            cv2.imshow('Frame', frame)

            if frames == 30:
                frames = 0

                current_result = self.gesture_recognition.recognize()
                if current_result != old_result:
                    if current_result == '[':
                        current_result = 'Ă'
                    elif current_result == ']':
                        current_result = 'Â'
                    print("Result: ", current_result)
                    old_result = current_result

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if keyboard.is_pressed('s'):
                ut.save_image_from_frame(mask, images_saved_count)
                images_saved_count += 1

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()


controller = Controller()
controller.run()
