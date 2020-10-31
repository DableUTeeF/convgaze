"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
import numpy as np
import os
import webcam_thread as cam

duration = 0.1  # seconds
freq = 440  # Hz


def save(loc):
    os.makedirs('output', exist_ok=True)
    global n
    cv2.imwrite(f'output/{loc}_{n}.jpg', raw)
    n += 1
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))


if __name__ == '__main__':
    images = []
    webcam = cam.WebcamThread(0, "Face detector 1", w=1280, h=720).start()
    n = 0
    if os.path.exists('output'):
        n = len(os.listdir('output')) // 2
    last_time = True
    cv2.namedWindow("d", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("d", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        # We get a new frame from the webcam
        _, frame = webcam.read()
        raw = frame.copy()
        image = frame.copy()
        try:
            # We send this frame to GazeTracking to analyze it

            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('q'):
                save(0)
            elif key == ord('w'):
                save(1)
            elif key == ord('e'):
                save(2)
            elif key == ord('r'):
                save(3)
            elif key == ord('t'):
                save(4)
            elif key == ord('y'):
                save(5)
            elif key == ord('a'):
                save(6)
            elif key == ord('s'):
                save(7)
            elif key == ord('d'):
                save(8)
            elif key == ord('f'):
                save(9)
            elif key == ord('g'):
                save(10)
            elif key == ord('h'):
                save(11)
            cv2.imshow('d', frame)
        except ZeroDivisionError:
            pass
