from keras import models, layers
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

def get_model():
    inpl = layers.Input((360, 640, 3))
    inpr = layers.Input((360, 640, 3))
    inpc = layers.Input((360, 640, 3))
    xl = layers.Conv2D(32, 3, strides=(2, 2))(inpl)
    xl = layers.BatchNormalization()(xl)
    xl = layers.Activation('relu')(xl)
    xl = layers.Conv2D(32, 3)(xl)
    xl = layers.BatchNormalization()(xl)
    xl = layers.Activation('relu')(xl)
    xl = layers.MaxPooling2D()(xl)

    xr = layers.Conv2D(32, 3, strides=(2, 2))(inpr)
    xr = layers.BatchNormalization()(xr)
    xr = layers.Activation('relu')(xr)
    xr = layers.Conv2D(32, 3)(xr)
    xr = layers.BatchNormalization()(xr)
    xr = layers.Activation('relu')(xr)
    xr = layers.MaxPooling2D()(xr)

    # xc = layers.Conv2D(32, 3, 2)(inpc)
    # xc = layers.BatchNormalization()(xc)
    # xc = layers.Activation('relu')(xc)
    # xc = layers.Conv2D(32, 3)(xc)
    # xc = layers.BatchNormalization()(xc)
    # xc = layers.Activation('relu')(xc)
    # xc = layers.MaxPooling2D()(xc)

    # x = layers.concatenate((xl, xc, xr))
    x = layers.concatenate([xl, xr])

    x = layers.Conv2D(96, 3, strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(512, 3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, 3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(2, activation='sigmoid')(x)

    # model = models.Model([inpl, inpc, inpr], x)
    model = models.Model([inpl, inpr], x)
    model.compile(optimizer='adam',
                  loss='mae')
    return model


def get_data():
    x1 = []
    x2 = []
    y = []
    for file in os.listdir('output/mouse/'):
        if 'jpg' in file:
            continue
        txt = open(os.path.join('output/mouse/', file)).read().split('\n')
        for line in txt:
            if len(line.split(',')) == 2:
                a, b = line.split(',')
                y.append(np.array((float(a) / 1920, float(b) / 1080)))
        im1 = cv2.imread(os.path.join('output/mouse/', file[:-4]+'_1.jpg'))
        im1 = cv2.resize(im1, (640, 360))
        im2 = cv2.imread(os.path.join('output/mouse/', file[:-4]+'_2.jpg'))
        im2 = cv2.resize(im2, (640, 360))
        # im3 = cv2.imread(os.path.join('output/mouse_old2/', file[:-4]+'_center.jpg'))
        # im3 = cv2.resize(im3, (640, 360))
        # ims = np.concatenate((im1, im2), axis=-1).astype('float32')
        # ims /= 127.5
        # ims -= 1
        # x.append(ims)

        x1.append(im1)
        x2.append(im2)
    return (np.array(x1, dtype='float32'), np.array(x2, dtype='float32')), np.array(y)


if __name__ == '__main__':
    (x1, x2), y = get_data()
    # X_train, X_test, y_train, y_test = train_test_split(x, y)

    model = get_model()
    model.summary()
    model.fit([x1, x2], y, epochs=4, batch_size=8)

    y_pred = model.predict(X_test)
    for i in range(y_test.shape[0]):
        print(f'y_true: {y_test[i]} - y_pred: {y_pred[i]}')

