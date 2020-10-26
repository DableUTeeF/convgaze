from regress_multi import get_data
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import cv2
import numpy as np
import os
from gaze_tracking import GazeTracking

if __name__ == '__main__':
    print('loaded')
    image = cv2.imread('/home/palm/Pictures/Screenshot from 2020-09-08 18-11-30.png')
    image = cv2.resize(image, (320, 540))
    raw_show = np.zeros((1080, 1920, 3), dtype='uint8')
    for i in range(6):
        for j in range(2):
            x = (i * 320, (i + 1) * 320)
            y = (j * 540, (j + 1) * 540)
            raw_show[y[0]:y[1], x[0]:x[1], :] = image

    x, y = get_data()
    model = MultiOutputRegressor(MLPRegressor())
    model.fit(x, y)
    print('trained')
    gaze1 = GazeTracking()
    gaze2 = GazeTracking()
    webcam1 = cv2.VideoCapture(2)
    webcam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    webcam2 = cv2.VideoCapture(4)
    webcam2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    detector = cv2.QRCodeDetector()
    cv2.namedWindow("d", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("d", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('d', raw_show)
    cv2.waitKey(1)
    while True:
        # We get a new frame from the webcam
        _, frame1 = webcam1.read()
        _, frame2 = webcam2.read()
        raw1 = frame1.copy()
        raw2 = frame2.copy()
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

        if left_pupil1 is not None and left_pupil2 is not None:
            xtest = [[gaze1.eye_left.center[0], gaze1.eye_left.center[1], gaze1.eye_right.center[0], gaze1.eye_right.center[1],
                     gaze1.pupil_left_coords()[0], gaze1.pupil_left_coords()[1], gaze1.pupil_right_coords()[0], gaze1.pupil_right_coords()[1],
                     gaze2.eye_left.center[0], gaze2.eye_left.center[1], gaze2.eye_right.center[0], gaze2.eye_right.center[1],
                     gaze2.pupil_left_coords()[0], gaze2.pupil_left_coords()[1], gaze2.pupil_right_coords()[0], gaze2.pupil_right_coords()[1],
                     ]]
            y_pred = model.predict(xtest)
            i, j = y_pred[0]
            print(i, j)
            loc_x = int(i * 1920 / 6)
            loc_y = int(j * 1080)
            dummy = raw_show.copy()
            cv2.circle(dummy, (loc_x, loc_y), 20, (0, 0, 255), -1)
            cv2.imshow('d', dummy)
            key = cv2.waitKey(1)
            if key == 27:
                break
