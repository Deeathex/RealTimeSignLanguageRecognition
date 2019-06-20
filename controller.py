import cv2

import gestures_recognition as gest_rec
import keyboard

import utils as ut


class Controller:
    def __init__(self):
        self.gesture_recognition = gest_rec.GestureRecognition('model_saved_2019-06-19.h5')

    def run(self):
        video_capture = cv2.VideoCapture(0)
        frames = 0
        images_saved_count = 0
        old_result = ""

        ret, last_frame = video_capture.read()
        last_frame_hand = self.gesture_recognition.get_hand_rectangle(last_frame)
        while video_capture.isOpened():
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            frames += 1

            hand_rectangle = self.gesture_recognition.get_hand_rectangle(frame)

            current_frame_hand = hand_rectangle
            motion = self.gesture_recognition.detect_motion(last_frame_hand, current_frame_hand, frames)
            last_frame_hand = current_frame_hand

            mask, contour = self.gesture_recognition.get_contour(hand_rectangle)

            # Display the resulting frame
            cv2.imshow('Video', hand_rectangle)
            cv2.imshow('Mask', mask)
            cv2.imshow('Frame', frame)

            test = cv2.bitwise_and(hand_rectangle, hand_rectangle, mask=mask)
            cv2.imshow('TEST', test)

            if frames == 30:
                frames = 0

                current_result = self.gesture_recognition.recognize()
                if current_result != old_result:
                    if current_result == '[':
                        current_result = 'Ă'
                    elif current_result == '\\':
                        current_result = 'Â'
                    elif (current_result == 'I') and motion:
                        current_result = 'Î'
                    elif ((current_result == 'S') or (current_result == 'E')) and motion:
                        current_result = 'Ș'
                    elif (current_result == 'T') and motion:
                        current_result = 'Ț'
                    elif ((current_result == 'Z') or (current_result == 'D')) and motion:
                        current_result = 'Z'
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
