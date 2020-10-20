from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os
import numpy as np


def get_data():
    x = []
    y = []
    for file in os.listdir('output/4 tiles'):
        if 'txt' in file:
            csv = open(os.path.join('output/4 tiles', file)).read().split('\n')
            y.append([int(file.split('_')[0])])
            data = []
            for line in [csv[0], csv[2], csv[3]]:
                l = line.split(',')
                data.extend([int(i) for i in l])
            x.append(data)
    return x, y


if __name__ == '__main__':
    x, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(mean_absolute_error(y_test, y_pred))
    for i in range(y_pred.shape[0]):
        print(y_test[i], y_pred[i])
