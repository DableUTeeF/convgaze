import dlib
from gaze_tracking.eye import Eye
from gaze_tracking.calibration import Calibration
import os
import cv2

if __name__ == '__main__':
    frame = cv2.imread('raw_screenshot_30.08.2020.png')
    roi = cv2.imread('roi_screenshot_30.08.2020.png')
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    model_path = os.path.abspath('gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat')
    predictor = dlib.shape_predictor(model_path)
    _face_detector = dlib.get_frontal_face_detector()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_detector(frame)
    landmarks = predictor(frame, 1)
