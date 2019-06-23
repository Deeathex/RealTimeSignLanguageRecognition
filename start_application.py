import cv2

from threading import Thread

import gestures_recognition as gest_rec
import keyboard

import utils as ut


class SignLanguageRecognition:
    def __init__(self):
        self.__gesture_recognition = gest_rec.GestureRecognition('model_saved_2019-06-19.h5')

    def run(self):
        video_capture = cv2.VideoCapture(0)
        frames = 0
        images_saved_count = 0
        current_result = ''
        old_result = ''

        is_voice_on = True

        ret, last_frame = video_capture.read()
        last_frame_hand = self.__gesture_recognition.get_hand_rectangle(last_frame)
        same_prediction = 0
        text = ''
        while video_capture.isOpened():
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            frames += 1

            hand_rectangle = self.__gesture_recognition.get_hand_rectangle(frame)

            current_frame_hand = hand_rectangle
            motion = self.__gesture_recognition.detect_motion(last_frame_hand, current_frame_hand, frames, thresh=10)
            last_frame_hand = current_frame_hand

            mask, contour = self.__gesture_recognition.get_mask_and_contour(hand_rectangle)

            self.__gesture_recognition.bounding_hand(contour)

            # Display the resulting frame
            frame_without_text = frame
            if len(text) == 51:
                frame = frame_without_text
                text = ''

            ut.show_blackboard_with_text(frame, text, is_voice_on)
            cv2.imshow('Video', hand_rectangle)
            cv2.imshow('Mask', mask)
            cv2.imshow('Frame', frame)

            test = cv2.bitwise_and(hand_rectangle, hand_rectangle, mask=mask)
            cv2.imshow('TEST', test)

            if frames == 30:
                frames = 0

                current_result = self.__gesture_recognition.recognize()
                if current_result != old_result:
                    same_prediction = 0
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
                else:
                    same_prediction += 1

            if same_prediction >= 5:
                text += current_result
                same_prediction = 0
                Thread(target=ut.say_text_romanian, args=(text, is_voice_on,)).start()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if keyboard.is_pressed('s'):
                ut.save_image_from_frame(mask, images_saved_count)
                images_saved_count += 1

            if keyboard.is_pressed('v') and is_voice_on:
                is_voice_on = False
            elif keyboard.is_pressed('v') and not is_voice_on:
                is_voice_on = True

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()


sign_language_recognition = SignLanguageRecognition()
sign_language_recognition.run()
