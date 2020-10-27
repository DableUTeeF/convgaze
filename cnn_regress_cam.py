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
    # xl = layers.MaxPooling2D()(xl)

    xr = layers.Conv2D(32, 3, strides=(2, 2))(inpr)
    xr = layers.BatchNormalization()(xr)
    xr = layers.Activation('relu')(xr)
    xr = layers.Conv2D(32, 3)(xr)
    xr = layers.BatchNormalization()(xr)
    xr = layers.Activation('relu')(xr)
    # xr = layers.MaxPooling2D()(xr)

    # xc = layers.Conv2D(32, 3, 2)(inpc)
    # xc = layers.BatchNormalization()(xc)
    # xc = layers.Activation('relu')(xc)
    # xc = layers.Conv2D(32, 3)(xc)
    # xc = layers.BatchNormalization()(xc)
    # xc = layers.Activation('relu')(xc)
    # xc = layers.MaxPooling2D()(xc)

    # x = layers.concatenate((xl, xc, xr))
    x = layers.concatenate([xl, xr])
    x = layers.MaxPooling2D()(x)
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
    [x1, x2], y = get_data()

    model = get_model()
    model.summary()
    model.fit([x1, x2], y, epochs=6, batch_size=8)

    print('trained')
    raw_show = np.zeros((1080, 1920, 3), dtype='uint8')
    webcam1 = cv2.VideoCapture(2)
    webcam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    webcam2 = cv2.VideoCapture(0)
    webcam2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    webcamcenter = cv2.VideoCapture(4)
    webcamcenter.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcamcenter.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    detector = cv2.QRCodeDetector()
    cv2.namedWindow("d", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("d", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('d', raw_show)
    cv2.waitKey(1)
    while True:
        # We get a new frame from the webcam
        _, frame1 = webcam1.read()
        _, frame2 = webcam2.read()
        _, framecenter = webcamcenter.read()
        im1 = cv2.resize(frame1, (640, 360))
        im2 = cv2.resize(frame2, (640, 360))
        ims = np.concatenate((im1, im2), axis=-1).astype('float32')
        # ims /= 127.5
        # ims -= 1
        im1 = np.array([im1], dtype='float32')
        im2 = np.array([im2], dtype='float32')
        y_pred = model.predict([im1, im2])
        i, j = y_pred[0]
        print(i, j)
        loc_x = int(i * 1920)
        loc_y = int(j * 1080)
        dummy = raw_show.copy()
        frame1 = cv2.resize(frame1, (640, 360))
        dummy[50:410, :640] = frame1
        frame2 = cv2.resize(frame2, (640, 360))
        dummy[500:860, :640] = frame2
        framecenter = cv2.resize(framecenter, (640, 360))
        dummy[50:410, 700:1340] = framecenter
        cv2.circle(dummy, (loc_x, loc_y), 20, (0, 0, 255), -1)
        cv2.imshow('d', dummy)
        key = cv2.waitKey(1)
        if key == 27:
            break

