from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os
import numpy as np


def get_data():
    x = []
    y = []
    for files in os.listdir('output/4 tiles'):
        if 'jpg' in files:
            continue
        obj = []
        cls, no = files.split('_')
        y.append(cls)
        txt = open(os.path.join('output/4 tiles', files)).read().split('\n')
        # eye_center = txt[1]
        # pupil_coords = txt[2]
        # box = txt[3]
        for line in txt[1:]:
            if line != '':
                a, b, c, d = line.split(',')
                obj.extend((float(a), float(b), float(c), float(d)))
        x.append(obj)
    return x, y


if __name__ == '__main__':
    x, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(x, y)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    z = np.round(y_pred)
    y_test = np.array(y_test, dtype='float64')
    print(mean_absolute_error(z, y_test))
    for i in range(y_test.shape[0]):
        print(f'y_true: {y_test[i]} - y_pred: {y_pred[i]} - rounded: {z[i]}')
