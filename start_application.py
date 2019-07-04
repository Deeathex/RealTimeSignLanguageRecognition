import cv2

from threading import Thread

import gestures_recognition as gest_rec
import keyboard

import utils as ut


class SignLanguageRecognition:
    def __init__(self):
        # self.__gesture_recognition = gest_rec.GestureRecognition('model_saved_2019-06-19.h5')
        # self.__gesture_recognition = gest_rec.GestureRecognition('LMGR_letters_space_newWord_valid=test/model_saved_2019-06-20.h5')
        self.__gesture_recognition = gest_rec.GestureRecognition('model_saved_2019-06-20.h5')

    def run(self):
        video_capture = cv2.VideoCapture(0)
        frames = 0
        images_saved_count = 0
        current_result = ''
        old_result = ''

        is_voice_on = True
        is_english_on = False

        ret, last_frame = video_capture.read()
        last_frame_hand = self.__gesture_recognition.get_hand_rectangle(last_frame)
        same_prediction = 0
        text = ''
        text_on_screen = ''
        word_finished = False
        current_result_is_with_motion = False

        key_v_was_pressed = False
        key_l_was_pressed = False
        key_s_was_pressed = False
        key_f_was_pressed = False

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
            if (len(text) == 51) or word_finished:
                frame = frame_without_text
                text = ''
                text_on_screen = ''
                word_finished = False

            ut.show_blackboard_with_text(frame, text_on_screen, is_voice_on)
            ut.show_blackboard_with_english_option(frame, is_english_on)
            cv2.imshow('Mask', mask)
            cv2.imshow('Frame', frame)

            test = cv2.bitwise_and(hand_rectangle, hand_rectangle, mask=mask)
            cv2.imshow('TEST', test)

            if frames == 30:
                frames = 0

                current_result = self.__gesture_recognition.recognize()
                current_result_is_with_motion = False
                if current_result == '[':
                    current_result = 'Ă'
                elif current_result == '\\':  # ] = Â
                    current_result = 'Â'
                elif (current_result == 'I') and motion:
                    current_result = 'Î'
                    current_result_is_with_motion = True
                elif ((current_result == 'S') or (current_result == 'E')) and motion:
                    current_result = 'Ș'
                    current_result_is_with_motion = True
                elif (current_result == 'T') and motion:
                    current_result = 'Ț'
                    current_result_is_with_motion = True
                elif ((current_result == 'Z') or (current_result == 'D')) and motion:
                    current_result = 'Z'
                    current_result_is_with_motion = True
                elif current_result == '^':  # _ = space
                    current_result = ' '
                elif current_result == ']':  # ^ = enter
                    word_finished = True
                    current_result = 'enter - new word'
                elif current_result == 'E':
                    current_result_is_with_motion = True

                if current_result != old_result:
                    same_prediction = 0
                    print("Result: ", current_result)
                    old_result = current_result
                else:
                    same_prediction += 1

            if same_prediction >= 3 or current_result_is_with_motion:
                text += current_result
                text_on_screen += ut.convert_to_printable(current_result)
                same_prediction = 0
                current_result_is_with_motion = False
                if is_english_on:
                    Thread(target=ut.say_text_english, args=(text, is_voice_on,)).start()
                else:
                    Thread(target=ut.say_text_romanian, args=(text, is_voice_on,)).start()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            key_s_is_pressed = keyboard.is_pressed('s')
            if key_s_is_pressed and not key_s_was_pressed:
                ut.save_image_from_frame(mask, images_saved_count)
                images_saved_count += 1
                key_s_was_pressed = True
            if key_s_was_pressed and not key_s_is_pressed:
                key_s_was_pressed = False

            key_v_is_pressed = keyboard.is_pressed('v')
            if key_v_is_pressed and not key_v_was_pressed:
                is_voice_on = not is_voice_on
                key_v_was_pressed = True
            if key_v_was_pressed and not key_v_is_pressed:
                key_v_was_pressed = False

            key_l_is_pressed = keyboard.is_pressed('l')
            if key_l_is_pressed and not key_l_was_pressed:
                is_english_on = not is_english_on
                key_l_was_pressed = True
            if key_l_was_pressed and not key_l_is_pressed:
                key_l_was_pressed = False

            key_f_is_pressed = keyboard.is_pressed('f')
            if key_f_is_pressed and not key_f_was_pressed:
                word_finished = True
                key_f_was_pressed = True
            if key_f_was_pressed and not key_f_is_pressed:
                key_f_was_pressed = False

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()


sign_language_recognition = SignLanguageRecognition()
sign_language_recognition.run()
