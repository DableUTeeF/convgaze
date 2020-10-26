from keras import layers, models
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

def get_model():
    model = models.Sequential(
        [
            # layers.Input(shape=(16,)),
            layers.Dense(256, activation='relu', input_shape=(16, )),
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


def get_data():
    x = []
    y = []
    for files in os.listdir('output_multicams/'):
        if 'jpg' in files:
            continue
        obj = []
        cls, no = files.split('_')
        cls = int(cls)
        j = cls // 6
        i = cls % 6

        y.append(np.array((i, j)))
        txt = open(os.path.join('output_multicams/', files)).read().split('\n')
        # eye_center = txt[1]
        # pupil_coords = txt[2]
        # box = txt[3]
        for line in txt:
            if line != '':
                a, b, c, d = line.split(',')
                obj.extend((float(a), float(b), float(c), float(d)))
        x.append(np.array(obj))
    return np.array(x, dtype='float32'), np.array(y)


if __name__ == '__main__':
    x, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(x, y)

    model = MultiOutputRegressor(SVR())
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    z = np.round(y_pred)
    y_test = np.array(y_test, dtype='float64')
    print(mean_absolute_error(z, y_test))
    for i in range(y_test.shape[0]):
        print(f'y_true: {y_test[i]} - y_pred: {y_pred[i]} - rounded: {z[i]}')


    # model = get_model()
    # model.fit(X_train, y_train, validation_data=[X_test, y_test], epochs=10)

