import cv2
import numpy as np


cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    image = frame.copy()
    # Display the resulting frame
    blue = frame[..., 0]
    green = frame[..., 1]
    red = frame[..., 2]

    mask = cv2.inRange(blue, 50, 255)
    output = cv2.bitwise_and(image, image, mask=mask)

    mask = cv2.inRange(red, 0, 50)  # todo: should be 100/150 with better plate
    output = cv2.bitwise_and(output, image, mask=mask)
    mask = cv2.inRange(green, 0, 80)
    output = cv2.bitwise_and(output, image, mask=mask)

    kernel = np.ones((5, 5), np.uint8)
    output = cv2.erode(output, kernel)

    output = cv2.dilate(output, kernel)

    cv2.imshow('filtered', output)
    cv2.imshow('raw', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # cv2.waitKey(1)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
