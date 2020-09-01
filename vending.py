import cv2
import os
import time
import numpy as np
images = []
pth = r'C:\Users\0\Downloads\PrototypePage\PrototypePage\Pics'
for files in os.listdir(pth):
    image = cv2.imread(pth)
    image = cv2.resize(image, (200, 300))
    images.append(image)
dummy = np.zeros((1200, 600))
while True:
    for i in range(6):
        for j in range(2):
            dummy[j*200:(j+1)*200, i*200:(i+1)*200] = images[np.random.randint(0, len(images)-1)]
    cv2.imshow('d', dummy)
    cv2.waitKey(1)
    time.sleep(2)
