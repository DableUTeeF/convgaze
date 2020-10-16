from regress import get_data
from sklearn.linear_model import LinearRegression
import cv2
import numpy as np
import os
from gaze_tracking import GazeTracking

if __name__ == '__main__':
    x, y = get_data()
    images = []
    pth = r'/home/root1/Downloads/PrototypePage/PrototypePage/Pics'
    for files in os.listdir(pth):
        image = cv2.imread(os.path.join(pth, files))
        image = cv2.resize(image, (320, 540))
        images.append(image)

    model = LinearRegression()
    model.fit(x, y)
    dummy = np.zeros((1080, 1920, 3), dtype='uint8')
    for i in range(6):
        for j in range(2):
            image = images[np.random.randint(0, len(images) - 1)]
            x = (i * 320, (i + 1) * 320)
            y = (j * 540, (j + 1) * 540)
            dummy[y[0]:y[1], x[0]:x[1], :] = image
    cv2.imshow('d', dummy)
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    gaze = GazeTracking()
    while True:
        # We get a new frame from the webcam
        _, frame = webcam.read()
        raw = frame.copy()
        image = frame.copy()
        gaze.refresh(frame)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()

        blue = frame[..., 0]
        green = frame[..., 1]
        red = frame[..., 2]
        mask1 = green > red * 1.7
        idx = (mask1 == 0)
        image[idx] = 0
        mask2 = green > blue * 1.1
        idx = (mask2 == 0)
        image[idx] = 0
        orif_filterd = image.copy()

        image = cv2.erode(image, np.ones((2, 2)))
        image = cv2.dilate(image, np.ones((2, 2)))
        image = cv2.dilate(image, np.ones((2, 2)))
        image = cv2.dilate(image, np.ones((2, 2)))
        image = cv2.dilate(image, np.ones((2, 2)))

        # mask = np.sum(image, axis=-1) > 0
        mask = image[..., 1] > 90
        mask = mask.astype('uint8') * 255

        _, contours, h = cv2.findContours(mask, 1, 2)
        xmin = None
        xmax = None
        ymin = None
        ymax = None
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            area = cv2.contourArea(cnt)
            if area < 2000:
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
        x = [[gaze.eye_left.center[0], gaze.eye_left.center[1], gaze.eye_right.center[0], gaze.eye_right.center[1],
              gaze.pupil_left_coords()[0], gaze.pupil_left_coords()[1], gaze.pupil_right_coords()[0], gaze.pupil_right_coords()[1],
              xmin, ymin, xmax, ymax]]
        y_pred = model.predict(x)
        z = np.round(y_pred)[0]
