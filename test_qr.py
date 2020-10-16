import cv2


if __name__ == '__main__':
    detector = cv2.QRCodeDetector()

    # img = cv2.imread('/home/root1/Pictures/Screenshot from 2020-09-11 21-20-28.png')
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    qr = cv2.imread('/home/root1/Pictures/nope_qr.png')
    while True:
        # We get a new frame from the webcam
        _, img = webcam.read()

        # img[100:300, 100:300] = qr
        # img[100:300, 400:600] = qr
        retval, points = detector.detectMulti(img)
        if points is not None:
            for qrcode in points:
                xmin = qrcode[0][0]
                xmax = qrcode[1][0]
                ymin = qrcode[0][1]
                ymax = qrcode[2][1]
                cv2.line(img,
                         (*qrcode[0],),
                         (*qrcode[1],), (0, 255, 0), 2)
                cv2.line(img,
                         (*qrcode[2],),
                         (*qrcode[3],), (0, 255, 0), 2)

        print(points)
        cv2.imshow('im', img)
        cv2.waitKey(1)
