"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import numpy as np
import os
import time

duration = 0.1  # seconds
freq = 440  # Hz


def save(loc):
    os.makedirs('output', exist_ok=True)
    global n, gaze, left_pupil
    if left_qr is None:
        return
    if left_pupil is None:
        return
    cv2.imwrite(f'output/{loc}_{n}.jpg', raw)
    with open(f'output/{loc}_{n}.txt', 'w') as wr:
        # wr.write(f'{left_pupil[0]},{left_pupil[1]},{right_pupil[0]},{right_pupil[1]}\n')
        wr.write(f'{gaze.eye_left.center[0]},{gaze.eye_left.center[1]},{gaze.eye_right.center[0]},{gaze.eye_right.center[1]}\n')
        wr.write(f'{gaze.pupil_left_coords()[0]},{gaze.pupil_left_coords()[1]},{gaze.pupil_right_coords()[0]},{gaze.pupil_right_coords()[1]}\n')
        wr.write(f'{left_qr[0][0]},{left_qr[0][1]},{left_qr[1][0]},{left_qr[1][1]}')
        wr.write(f'{left_qr[2][0]},{left_qr[2][1]},{left_qr[3][0]},{left_qr[3][1]}')
        wr.write(f'{right_qr[0][0]},{right_qr[0][1]},{right_qr[1][0]},{right_qr[1][1]}')
        wr.write(f'{right_qr[2][0]},{right_qr[2][1]},{right_qr[3][0]},{right_qr[3][1]}')
    n += 1
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))


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
    cv2.namedWindow("d", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("d", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    detector = cv2.QRCodeDetector()
    while True:
        # We get a new frame from the webcam
        _, frame = webcam.read()
        raw = frame.copy()
        image = frame.copy()
        try:
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

            retval, points = detector.detectMulti(frame)
            left_qr = []
            right_qr = []
            temp_qr = []
            if points is not None:
                for qrcode in points:
                    # topleft = qrcode[0]
                    # topright = qrcode[1]
                    # bottomright = qrcode[2]
                    # bottomleft = qrcode[3]
                    if len(temp_qr) == 0:
                        temp_qr = qrcode
                    else:
                        if qrcode[0][0] < temp_qr[0][0]:
                            right_qr = temp_qr
                            left_qr = qrcode
                        else:
                            left_qr = temp_qr
                            right_qr = qrcode
                    cv2.line(frame,
                             (*qrcode[0],),
                             (*qrcode[1],), (0, 0, 255), 8)
                    cv2.line(frame,
                             (*qrcode[2],),
                             (*qrcode[3],), (0, 0, 255), 8)

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
            # cv2.imshow("Demo", frame)
            if last_time:
                last_time = False
                for i in range(6):
                    for j in range(2):
                        image = images[np.random.randint(0, len(images) - 1)]
                        x = (i * 320, (i + 1) * 320)
                        y = (j * 540, (j + 1) * 540)
                        dummy[y[0]:y[1], x[0]:x[1], :] = image
            frame = cv2.resize(frame, (640, 360))
            cv2.line(frame, (320, 0), (320, 360), (0, 255, 255))
            dummy[50:410, :640] = frame
            cv2.imshow('d', dummy)
        except ZeroDivisionError:
            pass
