import cv2
import numpy as np


cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    image = frame.copy()
    ct = frame.copy()
    # Display the resulting frame
    blue = frame[..., 0]
    green = frame[..., 1]
    red = frame[..., 2]

    mask1 = blue > green
    idx = (mask1 == 0)
    image[idx] = 0

    mask2 = blue > red
    idx = (mask2 == 0)
    image[idx] = 0

    mask = np.sum(image, axis=-1) > 0
    mask = mask.astype('uint8') * 255

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 127, 255, 1)
    #
    # _, contours, h = cv2.findContours(thresh, 1, 2)
    _, contours, h = cv2.findContours(mask, 1, 2)
    for cnt in contours:

        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        # if area < 50:
        #     continue
        if len(approx) == 5:
            # print("pentagon")
            cv2.drawContours(ct, [cnt], 0, 255, -1)
        elif len(approx) == 3:
            # print("triangle")
            cv2.drawContours(ct, [cnt], 0, (0, 255, 0), -1)
        elif len(approx) == 4:
            # print("square")
            print(area)
            cv2.drawContours(ct, [cnt], 0, (0, 0, 255), -1)
        elif len(approx) == 9:
            # print("half-circle")
            cv2.drawContours(ct, [cnt], 0, (255, 255, 0), -1)
        elif len(approx) > 15:
            # print("circle")
            cv2.drawContours(ct, [cnt], 0, (0, 255, 255), -1)

    cv2.imshow('filtered', image)
    cv2.imshow('raw', frame)
    cv2.imshow('contour', ct)
    cv2.imshow('maskgreen', mask1.astype('uint8') * 255)
    cv2.imshow('maskred', mask2.astype('uint8') * 255)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # cv2.waitKey(1)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
