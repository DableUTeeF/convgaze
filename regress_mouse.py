from keras import layers, models
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import cv2
from gaze_tracking import GazeTracking


def get_data():
    x = []
    y = []
    gaze = GazeTracking()
    for files in os.listdir('output/mouse_old3/'):
        if 'jpg' in files:
            continue
        obj = []
        txt = open(os.path.join('output/mouse_old3/', files)).read().split('\n')
        center = cv2.imread(os.path.join('output/mouse_old3/', files[:-4] + '_center.jpg'))
        left = cv2.imread(os.path.join('output/mouse_old3/', files[:-4] + '_1.jpg'))
        right = cv2.imread(os.path.join('output/mouse_old3/', files[:-4] + '_2.jpg'))
        stop = False
        try:
            for im in [center, left, right]:
                gaze.refresh(im)
                if gaze.pupil_left_coords() is None or gaze.pupil_right_coords() is None:
                    stop = True
                    break
                obj.extend([gaze.eye_left.center[0], gaze.eye_left.center[1], gaze.eye_right.center[0], gaze.eye_right.center[1],
                            gaze.pupil_left_coords()[0], gaze.pupil_left_coords()[1], gaze.pupil_right_coords()[0], gaze.pupil_right_coords()[1], ])
            if stop:
                continue
            for line in txt:
                if len(line.split(',')) == 2:
                    a, b = line.split(',')
                    y.append(np.array((float(a) / 1920, float(b) / 1080)))
            x.append(np.array(obj))
        except cv2.error:
            pass
    return np.array(x, dtype='float32'), np.array(y)


def get_model():
    model = models.Sequential(
        [
            # layers.Input(shape=(16,)),
            layers.Dense(256, activation='relu', input_shape=(16,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(2, activation='sigmoid'),
        ]
    )
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['acc'],
    )
    return model


if __name__ == '__main__':
    x, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(x, y)

    model = MultiOutputRegressor(LinearRegression())
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    z = np.round(y_pred)
    y_test = np.array(y_test, dtype='float64')
    print(mean_absolute_error(z, y_test))
    for i in range(y_test.shape[0]):
        print(f'y_true: {y_test[i]} - y_pred: {y_pred[i]} - rounded: {z[i]}')

    # model = get_model()
    # model.fit(X_train, y_train, validation_data=[X_test, y_test], epochs=10)
