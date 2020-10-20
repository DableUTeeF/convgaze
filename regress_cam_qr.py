from regress import get_data
from sklearn.linear_model import LinearRegression
import cv2
import numpy as np
import os
from gaze_tracking import GazeTracking

if __name__ == '__main__':
    x, y = get_data()
    print('loaded')
    images = []
    pth = r'/home/root1/Downloads/PrototypePage/PrototypePage/Pics'
    for files in os.listdir(pth):
        image = cv2.imread(os.path.join(pth, files))
        image = cv2.resize(image, (320, 540))
        images.append(image)

    model = LinearRegression()
    model.fit(x, y)
    print('trained')

    raw_show = np.zeros((1080, 1920, 3), dtype='uint8')
    for i in range(6):
        for j in range(2):
            image = images[np.random.randint(0, len(images) - 1)]
            x = (i * 320, (i + 1) * 320)
            y = (j * 540, (j + 1) * 540)
            raw_show[y[0]:y[1], x[0]:x[1], :] = image
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    gaze = GazeTracking()
    detector = cv2.QRCodeDetector()
    # cv2.namedWindow("d", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("d", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('d', raw_show)
    cv2.waitKey(1)
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
        try:
            x = [[gaze.eye_left.center[0], gaze.eye_left.center[1], gaze.eye_right.center[0], gaze.eye_right.center[1],
                  gaze.pupil_left_coords()[0], gaze.pupil_left_coords()[1], gaze.pupil_right_coords()[0], gaze.pupil_right_coords()[1],
                  left_qr[0][0], left_qr[0][1], left_qr[1][0], left_qr[1][1],
                  left_qr[2][0], left_qr[2][1], left_qr[3][0], left_qr[3][1],
                  right_qr[0][0], right_qr[0][1], right_qr[1][0], right_qr[1][1],
                  right_qr[2][0], right_qr[2][1], right_qr[3][0], right_qr[3][1],
                  ]]
            y_pred = model.predict(x)
            z = np.round(y_pred)[0]
            print(z)
            j = z // 6
            i = z % 6
            x = (i * 320, (i + 1) * 320)
            y = (j * 540, (j + 1) * 540)
            dummy = raw_show.copy()
            dummy[y[0]:y[1], x[0]:x[1], 0] = 0
            dummy[y[0]:y[1], x[0]:x[1], 1] = 0
            dummy[y[0]:y[1], x[0]:x[1], 2] = 255
            frame = cv2.resize(frame, (640, 360))
            cv2.line(frame, (320, 0), (320, 360), (0, 255, 255))
        except:
            pass
