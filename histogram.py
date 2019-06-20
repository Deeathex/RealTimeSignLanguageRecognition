import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(
    'C:/Users/Deeathex/PycharmProjects/RealTimeSignLanguageRecognition/SavedImages/Back_Projection_Theory0.jpg', -1)
cv2.imshow('Photo', img)

color = ('b', 'g', 'r')
for channel, col in enumerate(color):
    histr = cv2.calcHist([img], [channel], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.title('Histogram for color scale picture')
plt.show()

print(histr)

while True:
    k = cv2.waitKey(0) & 0xFF
    if k == 27: break  # ESC key to exit
cv2.destroyAllWindows()
