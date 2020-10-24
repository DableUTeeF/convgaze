import cv2

if __name__ == '__main__':
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    n = 0
    while True:
        # We get a new frame from the webcam
        _, frame = webcam.read()
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('q'):
            cv2.imwrite(f'checkerboard/{n:02d}.jpg', frame)
            n += 1

