import cv2
import numpy as np
from gaze_tracking import GazeTracking
import os

def draw_events(event, x, y, flags, param):
    global raw1, raw2, n
    if event == cv2.EVENT_LBUTTONDOWN:
        if left_pupil1 is None:
            print('1 missing')
            return
        if left_pupil2 is None:
            print('2 missing')
            return
        cv2.imwrite(f'output/mouse/{n}_1.jpg', raw1)
        cv2.imwrite(f'output/mouse/{n}_2.jpg', raw2)
        with open(f'output/mouse/{n}.txt', 'w') as wr:
            # wr.write(f'{left_pupil[0]},{left_pupil[1]},{right_pupil[0]},{right_pupil[1]}\n')
            wr.write(f'{gaze1.eye_left.center[0]},{gaze1.eye_left.center[1]},{gaze1.eye_right.center[0]},{gaze1.eye_right.center[1]}\n')
            wr.write(f'{gaze1.pupil_left_coords()[0]},{gaze1.pupil_left_coords()[1]},{gaze1.pupil_right_coords()[0]},{gaze1.pupil_right_coords()[1]}\n')
            wr.write(f'{gaze2.eye_left.center[0]},{gaze2.eye_left.center[1]},{gaze2.eye_right.center[0]},{gaze2.eye_right.center[1]}\n')
            wr.write(f'{gaze2.pupil_left_coords()[0]},{gaze2.pupil_left_coords()[1]},{gaze2.pupil_right_coords()[0]},{gaze2.pupil_right_coords()[1]}\n')
            wr.write(f'{x},{y}')
        n += 1
        return

if __name__ == '__main__':
    raw_show = np.zeros((1080, 1920, 3), dtype='uint8')
    cv2.namedWindow("d", cv2.WND_PROP_FULLSCREEN)
    cv2.setMouseCallback('d', draw_events)
    cv2.setWindowProperty("d", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('d', raw_show)
    gaze1 = GazeTracking()
    gaze2 = GazeTracking()
    webcam1 = cv2.VideoCapture(2)
    webcam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    webcam2 = cv2.VideoCapture(4)
    webcam2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    n = len(os.listdir('/home/palm/PycharmProjects/true/output/mouse')) // 3
    while True:
        # We get a new frame from the webcam
        try:
            _, frame1 = webcam1.read()
            _, frame2 = webcam2.read()
            raw1 = frame1.copy()
            raw2 = frame2.copy()
            gaze1.refresh(frame1)
            frame1 = gaze1.annotated_frame()
            left_pupil1 = gaze1.pupil_left_coords()
            right_pupil1 = gaze1.pupil_right_coords()
            cv2.imshow('frame1', frame1)

            gaze2.refresh(frame2)
            frame2 = gaze2.annotated_frame()
            left_pupil2 = gaze2.pupil_left_coords()
            right_pupil2 = gaze2.pupil_right_coords()
            cv2.imshow('frame2', frame2)
        except cv2.error:
            pass
        key = cv2.waitKey(1)
        if key == 27:
            break

