import cv2
import os
import time
import numpy as np
images = []
pth = r'/home/root1/Downloads/PrototypePage/PrototypePage/Pics'
for files in os.listdir(pth):
    image = cv2.imread(os.path.join(pth, files))
    image = cv2.resize(image, (320, 540))
    images.append(image)
dummy = np.zeros((1080, 1920, 3), dtype='uint8')
while True:
    for i in range(6):
        for j in range(2):
            image = images[np.random.randint(0, len(images)-1)]
            x = (i*320, (i+1)*320)
            y = (j*540, (j+1)*540)
            dummy[y[0]:y[1], x[0]:x[1], :] = image
    cv2.imshow('d', dummy)
    k = cv2.waitKey(1)
    if k == 27:
        break
    time.sleep(2)
