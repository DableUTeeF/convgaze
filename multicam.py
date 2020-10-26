"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import numpy as np
import os
from keras import layers, models
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

def get_model(ty='sklear'):
    return MultiOutputRegressor(SVR())
    model = models.Sequential(
        [
            # layers.Input(shape=(16,)),
            layers.Dense(256, activation='relu', input_shape=(16, )),
            layers.Dense(256, activation='relu'),
            layers.Dense(2, activation='sigmoid'),
        ]
    )
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['acc'],
    )
    return model


def save(loc, write=True):
    os.makedirs('output', exist_ok=True)
    global n
    try:
        if left_pupil1 is None:
            return None, None
        if left_pupil2 is None:
            return None, None
        if write:
            cv2.imwrite(f'output_multicams/{loc}_{n}_1.jpg', raw1)
            cv2.imwrite(f'output_multicams/{loc}_{n}_2.jpg', raw2)
            with open(f'output_multicams/{loc}_{n}.txt', 'w') as wr:
                # wr.write(f'{left_pupil[0]},{left_pupil[1]},{right_pupil[0]},{right_pupil[1]}\n')
                wr.write(f'{gaze1.eye_left.center[0]},{gaze1.eye_left.center[1]},{gaze1.eye_right.center[0]},{gaze1.eye_right.center[1]}\n')
                wr.write(f'{gaze1.pupil_left_coords()[0]},{gaze1.pupil_left_coords()[1]},{gaze1.pupil_right_coords()[0]},{gaze1.pupil_right_coords()[1]}\n')
                wr.write(f'{gaze2.eye_left.center[0]},{gaze2.eye_left.center[1]},{gaze2.eye_right.center[0]},{gaze2.eye_right.center[1]}\n')
                wr.write(f'{gaze2.pupil_left_coords()[0]},{gaze2.pupil_left_coords()[1]},{gaze2.pupil_right_coords()[0]},{gaze2.pupil_right_coords()[1]}\n')
            n += 1
        x = [gaze1.eye_left.center[0], gaze1.eye_left.center[1], gaze1.eye_right.center[0], gaze1.eye_right.center[1],
             gaze1.pupil_left_coords()[0], gaze1.pupil_left_coords()[1], gaze1.pupil_right_coords()[0], gaze1.pupil_right_coords()[1],
             gaze2.eye_left.center[0], gaze2.eye_left.center[1], gaze2.eye_right.center[0], gaze2.eye_right.center[1],
             gaze2.pupil_left_coords()[0], gaze2.pupil_left_coords()[1], gaze2.pupil_right_coords()[0], gaze2.pupil_right_coords()[1],
             ]
        j = loc // 6
        i = loc % 6
        # print(i / 5, j / 2)
        return np.array(x), (i / 5, j)
    except IndexError:
        return None, None


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

    gaze1 = GazeTracking()
    gaze2 = GazeTracking()
    webcam1 = cv2.VideoCapture(2)
    webcam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    webcam2 = cv2.VideoCapture(4)
    webcam2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    n = 0
    if os.path.exists('output'):
        n = len(os.listdir('output')) // 2
    cv2.namedWindow("d", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("d", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    detector = cv2.QRCodeDetector()
    x = []
    y = []
    model = get_model()
    trained = False
    while True:
        # We get a new frame from the webcam
        _, frame1 = webcam1.read()
        _, frame2 = webcam2.read()
        raw1 = frame1.copy()
        raw2 = frame2.copy()
        try:
            gaze1.refresh(frame1)
            frame1 = gaze1.annotated_frame()
            left_pupil1 = gaze1.pupil_left_coords()
            right_pupil1 = gaze1.pupil_right_coords()
            # cv2.imshow('1', frame1)

            gaze2.refresh(frame2)
            frame2 = gaze2.annotated_frame()
            left_pupil2 = gaze2.pupil_left_coords()
            right_pupil2 = gaze2.pupil_right_coords()
            # cv2.imshow('2', frame2)

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
            elif key == 32:
                model = get_model()
                model.fit(np.array(x), np.array(y), epochs=2)
                trained = True
            if x1 is not None:
                x.append(x1)
                y.append(y1)
            if trained and left_pupil1 is not None and left_pupil2 is not None:
                xtest = [gaze1.eye_left.center[0], gaze1.eye_left.center[1], gaze1.eye_right.center[0], gaze1.eye_right.center[1],
                          gaze1.pupil_left_coords()[0], gaze1.pupil_left_coords()[1], gaze1.pupil_right_coords()[0], gaze1.pupil_right_coords()[1],
                          gaze2.eye_left.center[0], gaze2.eye_left.center[1], gaze2.eye_right.center[0], gaze2.eye_right.center[1],
                          gaze2.pupil_left_coords()[0], gaze2.pupil_left_coords()[1], gaze2.pupil_right_coords()[0], gaze2.pupil_right_coords()[1],
                          ]
                y_pred = model.predict(xtest)
                i, j = np.round(y_pred)[0]
                print(i, j)
                loc_x = int(i * 1280)
                loc_y = int(j * 720)
                dummy = raw_show.copy()
                cv2.circle(dummy, (loc_x, loc_y), 20, (0, 0, 255), -1)
                cv2.imshow('d', dummy)
            else:
                dummy = raw_show.copy()
                frame1 = cv2.resize(frame1, (640, 360))
                cv2.line(frame1, (320, 0), (320, 360), (0, 255, 255))
                dummy[50:410, :640] = frame1
                frame2 = cv2.resize(frame2, (640, 360))
                cv2.line(frame2, (320, 0), (320, 360), (0, 255, 255))
                dummy[500:860, :640] = frame2
                cv2.imshow('d', dummy)

        except ZeroDivisionError:
            pass
        except cv2.error:
            pass
