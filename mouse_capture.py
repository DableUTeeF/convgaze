import cv2
import numpy as np
from gaze_tracking import GazeTracking
import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import time
import webcam_thread as cam

camera_matrix = np.array([[768.41339356, 0., 287.08190938], [0., 769.01503164, 240.83347034], [0., 0., 1.]])
dist_coefs = np.array([0.02291718, -0.54353385, 0.00239008, 0.01015446, 0.97942389])

def draw_events(event, x, y, flags, param):
    global raw1, n, samples, targets
    if event == cv2.EVENT_LBUTTONDOWN:
        # if left_pupil1 is None:
        #     print('1 missing')
        #     return
        # if left_pupil2 is None:
        #     print('2 missing')
        #     return
        cv2.imwrite(f'output/mouse/{n}_{x//320}_{y//240}.jpg', raw1)
        with open(f'output/mouse/{n}_{x//320}_{y//240}.txt', 'w') as wr:
            # wr.write(f'{gaze1.eye_left.center[0]},{gaze1.eye_left.center[1]},{gaze1.eye_right.center[0]},{gaze1.eye_right.center[1]}\n')
            # wr.write(f'{gaze1.pupil_left_coords()[0]},{gaze1.pupil_left_coords()[1]},{gaze1.pupil_right_coords()[0]},{gaze1.pupil_right_coords()[1]}\n')
            # wr.write(f'{gaze2.eye_left.center[0]},{gaze2.eye_left.center[1]},{gaze2.eye_right.center[0]},{gaze2.eye_right.center[1]}\n')
            # wr.write(f'{gaze2.pupil_left_coords()[0]},{gaze2.pupil_left_coords()[1]},{gaze2.pupil_right_coords()[0]},{gaze2.pupil_right_coords()[1]}\n')
            # wr.write(f'{gazecenter.eye_left.center[0]},{gazecenter.eye_left.center[1]},{gazecenter.eye_right.center[0]},{gazecenter.eye_right.center[1]}\n')
            # wr.write(f'{gazecenter.pupil_left_coords()[0]},{gazecenter.pupil_left_coords()[1]},{gazecenter.pupil_right_coords()[0]},{gazecenter.pupil_right_coords()[1]}\n')
            wr.write(f'{x},{y}')
        a = x
        b = y
        target = np.zeros((360, 640, 1))
        target[max(0, b - 31):b + 30, max(0, a - 31):a + 30] += 0.25
        target[max(0, b - 15):b + 14, max(0, a - 15):a + 14] += 0.25
        target[max(0, b - 7):b + 6, max(0, a - 7):a + 6] += 0.25
        target[max(0, b - 3):b + 2, max(0, a - 3):a + 2] += 0.25
        cv2.imwrite(f'output/mouse/{n}_{x//320}_{y//240}_target.jpg', (target * 255).astype('uint8'))
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
    webcam1 = cam.WebcamThread(0, "Face detector 1").start()

    # webcam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # webcam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    n = len(os.listdir('/home/palm/PycharmProjects/true/output/mouse')) // 3
    samples = []
    targets = []
    trained = False
    while True:
        # We get a new frame from the webcam
        t = time.time()
        try:
            _, frame1 = webcam1.read()
            h, w = frame1.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

            frame1 = cv2.undistort(frame1, camera_matrix, dist_coefs, None, newcameramtx)
            # crop and save the image
            x, y, w, h = roi
            frame1 = frame1[y:y + h, x:x + w]
            frame1 = cv2.resize(frame1, (640, 473))
            frame1 = frame1[:360, :]

            t1 = time.time()
            raw1 = frame1.copy()

            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == 32:
                model = MultiOutputRegressor(LinearRegression())
                model.fit(np.array(samples), np.array(targets))
                trained = True
            dummy = frame1.copy()

            dummy = cv2.putText(dummy, f'{(t1 - t) * 1000} ms', (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.1, (147, 58, 31), 2)
            cv2.imshow('d', dummy)

        except cv2.error:
            pass
