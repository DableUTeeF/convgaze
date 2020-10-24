import cv2
import numpy as np
from gaze_tracking import GazeTracking

gaze = GazeTracking()
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    ret, frame = cap.read()
    image = frame.copy()
    ct = frame.copy()

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
    for cnt in contours:

        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if area < 2000 : #or len(approx) != 4:
            continue
        print(len(approx))
        x = approx[..., 0]
        y = approx[..., 1]
        xmin = min(x)[0]
        xmax = max(x)[0]
        ymin = min(y)[0]
        ymax = max(y)[0]
        # cv2.drawContours(ct, [cnt], 0, 255, 2)
        cv2.rectangle(ct,
                      (xmin, ymin),
                      (xmax, ymax), (0, 255, 0), 2)

    cv2.imshow('filtered', image)
    cv2.imshow('orif_filterd', orif_filterd)
    cv2.imshow('raw', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('contour', ct)
    cv2.imshow('maskgreen', mask1.astype('uint8') * 255)
    cv2.imshow('maskred', mask2.astype('uint8') * 255)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # cv2.waitKey(1)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
