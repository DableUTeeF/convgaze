"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import numpy as np
import os
import time

def save(loc):
    os.makedirs('output', exist_ok=True)
    global n
    cv2.imwrite(f'output/{loc}_{n}.jpg', raw)
    with open(f'output/{loc}_{n}.txt', 'w') as wr:
        wr.write(f'{left_pupil[0]},{left_pupil[1]},{right_pupil[0]},{right_pupil[1]}\n')
        wr.write(
            f'{gaze.eye_left.center[0]},{gaze.eye_left.center[1]},{gaze.eye_right.center[0]},{gaze.eye_right.center[1]}\n')
        wr.write(f'{xmin},{ymin},{xmax},{ymax}')
    n += 1


if __name__ == '__main__':
    images = []
    pth = r'/home/root1/Downloads/PrototypePage/PrototypePage/Pics'
    for files in os.listdir(pth):
        image = cv2.imread(os.path.join(pth, files))
        image = cv2.resize(image, (320, 540))
        images.append(image)
    dummy = np.zeros((1080, 1920, 3), dtype='uint8')

    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    n = 0
    if os.path.exists('output'):
        n = len(os.listdir('output')) // 2
    last_time = True
    while True:
        # We get a new frame from the webcam
        _, frame = webcam.read()
        raw = frame.copy()
        image = frame.copy()

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)

        frame = gaze.annotated_frame()
        text = ""

        if gaze.is_blinking():
            text = "Blinking"
        elif gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"

        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()

        blue = frame[..., 0]
        green = frame[..., 1]
        red = frame[..., 2]
        mask1 = green > red * 2.5
        idx = (mask1 == 0)
        image[idx] = 0
        mask2 = green > blue * 1.1
        idx = (mask2 == 0)
        image[idx] = 0
        mask = np.sum(image, axis=-1) > 0
        mask = mask.astype('uint8') * 255

        _, contours, h = cv2.findContours(mask, 1, 2)
        xmin = None
        xmax = None
        ymin = None
        ymax = None
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            area = cv2.contourArea(cnt)
            if area < 20000:
                continue
            x = approx[..., 0]
            y = approx[..., 1]
            xmin = min(x)[0]
            xmax = max(x)[0]
            ymin = min(y)[0]
            ymax = max(y)[0]
            # cv2.drawContours(ct, [cnt], 0, 255, 2)
            cv2.rectangle(frame,
                          (xmin, ymin),
                          (xmax, ymax), (0, 255, 0), 2)
            break

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('q'):
            save(0)
        elif key == ord('w'):
            save(1)
        elif key == ord('e'):
            save(2)
        elif key == ord('r'):
            save(3)
        elif key == ord('t'):
            save(4)
        elif key == ord('y'):
            save(5)
        elif key == ord('a'):
            save(6)
        elif key == ord('s'):
            save(7)
        elif key == ord('d'):
            save(8)
        elif key == ord('f'):
            save(9)
        elif key == ord('g'):
            save(10)
        elif key == ord('h'):
            save(11)
        cv2.imshow("Demo", frame)
        if last_time:
            last_time = False
            for i in range(6):
                for j in range(2):
                    image = images[np.random.randint(0, len(images) - 1)]
                    x = (i * 320, (i + 1) * 320)
                    y = (j * 540, (j + 1) * 540)
                    dummy[y[0]:y[1], x[0]:x[1], :] = image
        cv2.imshow('d', dummy)


