import cv2
import numpy as np


def video_capture():
    cap = cv2.VideoCapture(0)  # You can replace it with your video path
    i = 0
    while True:
        ret, frame = cap.read()
        if frame is not None:
            i = i + 1
            cv2.imwrite('results/daisy.1.jpg', frame, (299, 299))
            cv2.imshow("Cropped_Face", frame)
            if i == 2:
                break

    cap.release()
    cv2.destroyAllWindows()