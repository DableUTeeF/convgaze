import cv2
import numpy as np
from gaze_tracking import GazeTracking
import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import time


def draw_events(event, x, y, flags, param):
    global raw1, raw2, rawcenter, n, samples, targets
    if event == cv2.EVENT_LBUTTONDOWN:
        # if left_pupil1 is None:
        #     print('1 missing')
        #     return
        # if left_pupil2 is None:
        #     print('2 missing')
        #     return
        cv2.imwrite(f'output/mouse/{n}_1.jpg', raw1)
        cv2.imwrite(f'output/mouse/{n}_2.jpg', raw2)
        cv2.imwrite(f'output/mouse/{n}_center.jpg', rawcenter)
        with open(f'output/mouse/{n}.txt', 'w') as wr:
            # wr.write(f'{gaze1.eye_left.center[0]},{gaze1.eye_left.center[1]},{gaze1.eye_right.center[0]},{gaze1.eye_right.center[1]}\n')
            # wr.write(f'{gaze1.pupil_left_coords()[0]},{gaze1.pupil_left_coords()[1]},{gaze1.pupil_right_coords()[0]},{gaze1.pupil_right_coords()[1]}\n')
            # wr.write(f'{gaze2.eye_left.center[0]},{gaze2.eye_left.center[1]},{gaze2.eye_right.center[0]},{gaze2.eye_right.center[1]}\n')
            # wr.write(f'{gaze2.pupil_left_coords()[0]},{gaze2.pupil_left_coords()[1]},{gaze2.pupil_right_coords()[0]},{gaze2.pupil_right_coords()[1]}\n')
            # wr.write(f'{gazecenter.eye_left.center[0]},{gazecenter.eye_left.center[1]},{gazecenter.eye_right.center[0]},{gazecenter.eye_right.center[1]}\n')
            # wr.write(f'{gazecenter.pupil_left_coords()[0]},{gazecenter.pupil_left_coords()[1]},{gazecenter.pupil_right_coords()[0]},{gazecenter.pupil_right_coords()[1]}\n')
            wr.write(f'{x},{y}')
        a = x // 3
        b = y // 3
        target = np.zeros((384, 640, 1))
        target[max(0, b - 31):b + 30, max(0, a - 31):a + 30] += 0.25
        target[max(0, b - 15):b + 14, max(0, a - 15):a + 14] += 0.25
        target[max(0, b - 7):b + 6, max(0, a - 7):a + 6] += 0.25
        target[max(0, b - 3):b + 2, max(0, a - 3):a + 2] += 0.25
        cv2.imwrite(f'output/mouse/{n}_target.jpg', (target * 255).astype('uint8'))
        n += 1
        # sample = [gaze1.eye_left.center[0], gaze1.eye_left.center[1], gaze1.eye_right.center[0], gaze1.eye_right.center[1],
        #           gaze1.pupil_left_coords()[0], gaze1.pupil_left_coords()[1], gaze1.pupil_right_coords()[0], gaze1.pupil_right_coords()[1],
        #           gaze2.eye_left.center[0], gaze2.eye_left.center[1], gaze2.eye_right.center[0], gaze2.eye_right.center[1],
        #           gaze2.pupil_left_coords()[0], gaze2.pupil_left_coords()[1], gaze2.pupil_right_coords()[0], gaze2.pupil_right_coords()[1],
        #           gazecenter.eye_left.center[0], gazecenter.eye_left.center[1], gazecenter.eye_right.center[0], gazecenter.eye_right.center[1],
        #           gazecenter.pupil_left_coords()[0], gazecenter.pupil_left_coords()[1], gazecenter.pupil_right_coords()[0], gazecenter.pupil_right_coords()[1],
        #           ]
        # samples.append(np.array(sample, dtype='float32'))
        # targets.append(np.array([x / 1920, y / 1080], dtype='float32'))
        return


if __name__ == '__main__':
    raw_show = np.zeros((1080, 1920, 3), dtype='uint8')
    cv2.namedWindow("d", cv2.WND_PROP_FULLSCREEN)
    cv2.setMouseCallback('d', draw_events)
    cv2.setWindowProperty("d", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('d', raw_show)
    gaze1 = GazeTracking()
    gaze2 = GazeTracking()
    gazecenter = GazeTracking()
    webcam1 = cv2.VideoCapture(2)
    webcam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    webcam2 = cv2.VideoCapture(0)
    webcam2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    webcamcenter = cv2.VideoCapture(4)
    webcamcenter.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcamcenter.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    n = len(os.listdir('/home/palm/PycharmProjects/true/output/mouse')) // 5
    samples = []
    targets = []
    trained = False
    while True:
        # We get a new frame from the webcam
        t = time.time()
        try:
            _, frame1 = webcam1.read()
            _, frame2 = webcam2.read()
            _, framecenter = webcamcenter.read()
            t1 = time.time()
            raw1 = frame1.copy()
            raw2 = frame2.copy()
            rawcenter = framecenter.copy()
            # gaze1.refresh(frame1)
            # frame1 = gaze1.annotated_frame()
            # left_pupil1 = gaze1.pupil_left_coords()
            # right_pupil1 = gaze1.pupil_right_coords()
            # cv2.imshow('frame1', frame1)

            # gaze2.refresh(frame2)
            # frame2 = gaze2.annotated_frame()
            # left_pupil2 = gaze2.pupil_left_coords()
            # right_pupil2 = gaze2.pupil_right_coords()
            # cv2.imshow('frame2', frame2)

            # gazecenter.refresh(framecenter)
            # framecenter = gazecenter.annotated_frame()
            # left_pupilcenter = gazecenter.pupil_left_coords()
            # right_pupilcenter = gazecenter.pupil_right_coords()
            # cv2.imshow('framecenter', framecenter)

            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == 32:
                model = MultiOutputRegressor(LinearRegression())
                model.fit(np.array(samples), np.array(targets))
                trained = True
            dummy = raw_show.copy()
            frame1 = cv2.resize(frame1, (640, 360))
            dummy[50:410, :640] = frame1
            frame2 = cv2.resize(frame2, (640, 360))
            dummy[500:860, :640] = frame2
            framecenter = cv2.resize(framecenter, (640, 360))
            dummy[50:410, 700:1340] = framecenter

            if trained:
                xtest = [[gaze1.eye_left.center[0], gaze1.eye_left.center[1], gaze1.eye_right.center[0], gaze1.eye_right.center[1],
                          gaze1.pupil_left_coords()[0], gaze1.pupil_left_coords()[1], gaze1.pupil_right_coords()[0], gaze1.pupil_right_coords()[1],
                          gaze2.eye_left.center[0], gaze2.eye_left.center[1], gaze2.eye_right.center[0], gaze2.eye_right.center[1],
                          gaze2.pupil_left_coords()[0], gaze2.pupil_left_coords()[1], gaze2.pupil_right_coords()[0], gaze2.pupil_right_coords()[1],
                          gazecenter.eye_left.center[0], gazecenter.eye_left.center[1], gazecenter.eye_right.center[0], gazecenter.eye_right.center[1],
                          gazecenter.pupil_left_coords()[0], gazecenter.pupil_left_coords()[1], gazecenter.pupil_right_coords()[0], gazecenter.pupil_right_coords()[1],
                          ]]
                y_pred = model.predict(xtest)
                print(y_pred[0])
                i, j = y_pred[0]
                loc_x = int(i * 1920)
                loc_y = int(j * 1080)
                cv2.circle(dummy, (loc_x, loc_y), 20, (0, 0, 255), -1)
            dummy = cv2.putText(dummy, f'{(t1 - t) * 1000} ms', (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
            cv2.imshow('d', dummy)

        except cv2.error:
            pass
