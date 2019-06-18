import cv2
from tensorflow.python.keras.models import load_model
import numpy as np
from tensorflow.python.keras.preprocessing import image
import utils as ut

video_capture = cv2.VideoCapture(0)

# import the trained model
classifier = load_model('model_saved_2019-06-18.h5')
oldResult = ""

frames = 0
k = 0
while video_capture.isOpened():
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frames += 1

    # # Draw a rectangle
    # for (x, y, w, h) in faces:
    x = 50
    y = 100
    w = 200
    h = 200
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

    hand_rectangle = frame[y:y + h, x:x + w]

    # aplying mask and using hand segmentation to find contours
    image_ycrcb = cv2.cvtColor(hand_rectangle, cv2.COLOR_BGR2YCR_CB)
    blur = cv2.GaussianBlur(image_ycrcb, (11, 11), 0)

    skin_ycrcb_min = np.array((0, 50, 67))
    skin_ycrcb_max = np.array((255, 173, 130))
    mask = cv2.inRange(blur, skin_ycrcb_min,
                       skin_ycrcb_max)  # detecting the hand in the bounding box using skin detection

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 2)
    contour = ut.get_max_contour(contours, 4000)  # using contours to capture the skin filtered image of the hand

    if contour is not None and contour.all is not None:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(hand_rectangle, (x, y), (x + w, y + h), (0, 255, 0), 2)
        hand_with_contour = hand_rectangle[y:y + h, x:x + w]
        hand_with_contour = cv2.bitwise_and(hand_with_contour, hand_with_contour, mask=mask[y:y + h, x:x + w])
        hand_with_contour_gray = cv2.cvtColor(hand_with_contour, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(hand_with_contour_gray, (5, 5), 0)
        ret_otsu_threshold, otsu_threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.drawContours(otsu_threshold, contours, -1, (255, 255, 255), -1)
        otsu_threshold = cv2.resize(otsu_threshold, (200, 200))

    # Display the resulting frame
    cv2.imshow('Video', hand_rectangle)
    cv2.imshow('Mask', mask)

    if frames == 30:
        frames = 0

        image_to_be_saved = cv2.resize(mask, (50, 50))
        cv2.imwrite('C:/Users/Deeathex/PycharmProjects/TestSignLang1/SavedImages/predict.JPG', image_to_be_saved)

        test_image = image.load_img('C:/Users/Deeathex/PycharmProjects/TestSignLang1/SavedImages/predict.JPG',
                                    color_mode="grayscale")
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = classifier.predict(test_image)

        max = -1
        position = -1
        i = 0
        for probability in result[0]:
            if probability > max:
                max = probability
                position = i
            i += 1

        currentResult = chr(position + 65)
        if currentResult != oldResult:
            if (currentResult == '['):
                currentResult = 'Ă'
            elif currentResult == ']':
                currentResult = 'Â'
            print("Result: ", currentResult)
            oldResult = currentResult

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    import keyboard

    if keyboard.is_pressed('s'):
        image_to_be_saved = cv2.resize(mask, (50, 50))
        cv2.imwrite('C:/Users/Deeathex/PycharmProjects/TestSignLang1/SavedImages/' + str(k) + '.JPG', image_to_be_saved)
        k += 1
    if k == 1250:
        print("STOP: 1250 images saved!")

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
