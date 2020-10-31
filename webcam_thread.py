from threading import Thread
import cv2


class WebcamThread:

    def __init__(self, src=0, name="WebcamThread", af=None, f=None, w=None, h=None):
        self.cap = cv2.VideoCapture(src)
        if af is not None:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if f is not None:
            self.cap.set(cv2.CAP_PROP_FOCUS, 0)
        if w is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        if h is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        _, self.frame = self.cap.read()
        self.name = name
        self.stopped = False

    def update(self):
        while True:
            if self.stopped:
                return
            _, self.frame = self.cap.read()

    def start(self):
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def read(self):
        return None, self.frame

    def stop(self):
        self.stopped = True
