"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import numpy as np
import os
from sklearn.linear_model import LinearRegression

duration = 0.1  # seconds
freq = 440  # Hz


def save(loc):
    os.makedirs('output', exist_ok=True)
    global n, gaze, left_pupil
    try:
        if len(left_qr) == 0:
            return None, None
        if len(right_qr) == 0:
            return None, None
        if left_pupil is None:
            return None, None
        cv2.imwrite(f'output/{loc}_{n}.jpg', raw)
        with open(f'output/{loc}_{n}.txt', 'w') as wr:
            # wr.write(f'{left_pupil[0]},{left_pupil[1]},{right_pupil[0]},{right_pupil[1]}\n')
            wr.write(f'{gaze.eye_left.center[0]},{gaze.eye_left.center[1]},{gaze.eye_right.center[0]},{gaze.eye_right.center[1]}\n')
            wr.write(f'{gaze.pupil_left_coords()[0]},{gaze.pupil_left_coords()[1]},{gaze.pupil_right_coords()[0]},{gaze.pupil_right_coords()[1]}\n')
            wr.write(f'{left_qr[0][0]},{left_qr[0][1]},{left_qr[1][0]},{left_qr[1][1]}\n')
            wr.write(f'{left_qr[2][0]},{left_qr[2][1]},{left_qr[3][0]},{left_qr[3][1]}\n')
            wr.write(f'{right_qr[0][0]},{right_qr[0][1]},{right_qr[1][0]},{right_qr[1][1]}\n')
            wr.write(f'{right_qr[2][0]},{right_qr[2][1]},{right_qr[3][0]},{right_qr[3][1]}\n')
        n += 1
        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
        x = [gaze.eye_left.center[0], gaze.eye_left.center[1], gaze.eye_right.center[0], gaze.eye_right.center[1],
              gaze.pupil_left_coords()[0], gaze.pupil_left_coords()[1], gaze.pupil_right_coords()[0], gaze.pupil_right_coords()[1],
              left_qr[0][0], left_qr[0][1], left_qr[1][0], left_qr[1][1],
              left_qr[2][0], left_qr[2][1], left_qr[3][0], left_qr[3][1],
              right_qr[0][0], right_qr[0][1], right_qr[1][0], right_qr[1][1],
              right_qr[2][0], right_qr[2][1], right_qr[3][0], right_qr[3][1],
              ]
        y = loc
        return x, y
    except IndexError:
        x = None
        y = None
        return x, y


if __name__ == '__main__':
    images = []
    image = cv2.imread('/home/palm/Pictures/Screenshot from 2020-09-08 18-11-30.png')
    image = cv2.resize(image, (320, 540))
    raw_show = np.zeros((1080, 1920, 3), dtype='uint8')
    for i in range(6):
        for j in range(2):
            x = (i * 320, (i + 1) * 320)
            y = (j * 540, (j + 1) * 540)
            raw_show[y[0]:y[1], x[0]:x[1], :] = image

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
    x = []
    y = []
    model = LinearRegression()
    trained = False
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

            x1, y1 = None, None
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('q'):
                x1, y1 = save(0)
            elif key == ord('w'):
                x1, y1 = save(1)
            elif key == ord('e'):
                x1, y1 = save(2)
            elif key == ord('r'):
                x1, y1 = save(3)
            elif key == ord('t'):
                x1, y1 = save(4)
            elif key == ord('y'):
                x1, y1 = save(5)
            elif key == ord('a'):
                x1, y1 = save(6)
            elif key == ord('s'):
                x1, y1 = save(7)
            elif key == ord('d'):
                x1, y1 = save(8)
            elif key == ord('f'):
                x1, y1 = save(9)
            elif key == ord('g'):
                x1, y1 = save(10)
            elif key == ord('h'):
                x1, y1 = save(11)
            elif key == ord('l'):
                model = LinearRegression()
                model.fit(x, y)
                trained = True
            if x1 is not None:
                x.append(x1)
                y.append(y1)
            if trained and len(left_qr) > 0 and left_pupil is not None:
                    xtest = [[gaze.eye_left.center[0], gaze.eye_left.center[1], gaze.eye_right.center[0], gaze.eye_right.center[1],
                          gaze.pupil_left_coords()[0], gaze.pupil_left_coords()[1], gaze.pupil_right_coords()[0], gaze.pupil_right_coords()[1],
                          left_qr[0][0], left_qr[0][1], left_qr[1][0], left_qr[1][1],
                          left_qr[2][0], left_qr[2][1], left_qr[3][0], left_qr[3][1],
                          right_qr[0][0], right_qr[0][1], right_qr[1][0], right_qr[1][1],
                          right_qr[2][0], right_qr[2][1], right_qr[3][0], right_qr[3][1],
                          ]]
                    y_pred = model.predict(xtest)
                    z = np.round(y_pred)[0]
                    print(z)
                    j = z // 6
                    i = z % 6
                    
                    loc_x = (int(i * 320), int((i + 1) * 320))
                    loc_y = (int(j * 540), int((j + 1) * 540))
                    dummy = raw_show.copy()
                    dummy[loc_y[0]:loc_y[1], loc_x[0]:loc_x[1], 0] = 0
                    dummy[loc_y[0]:loc_y[1], loc_x[0]:loc_x[1], 1] = 0
                    dummy[loc_y[0]:loc_y[1], loc_x[0]:loc_x[1], 2] = 255
                    frame = cv2.resize(frame, (640, 360))
                    cv2.line(frame, (320, 0), (320, 360), (0, 255, 255))
                    cv2.imshow('d', dummy)
            else:
                dummy = raw_show.copy()
                frame = cv2.resize(frame, (640, 360))
                cv2.line(frame, (320, 0), (320, 360), (0, 255, 255))
                dummy[50:410, :640] = frame
                cv2.imshow('d', dummy)
        except ZeroDivisionError:
            pass
