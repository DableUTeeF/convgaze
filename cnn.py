from keras import models, layers, optimizers
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision.models import mobilenet_v2
import natthaphon
from torch.functional import F

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bacabone = mobilenet_v2(pretrained=True).features
        self.output = nn.Linear(1280, 4)

    def forward(self, x):
        x = self.bacabone(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.output(x)
        return x

    @staticmethod
    def loss(outputs, labels):
        outputs = outputs.reshape(outputs.shape[0], -1)
        labels = labels.reshape(labels.shape[0], -1)
        loss = F.cross_entropy(outputs.double(), labels.double().max(1)[1])
        return loss


def get_model():
    inpl = layers.Input((360, 640, 3))
    inpr = layers.Input((360, 640, 3))
    inpc = layers.Input((384, 640, 3))
    # xl = layers.Conv2D(32, 3, strides=(2, 2))(inpl)
    # xl = layers.BatchNormalization()(xl)
    # xl = layers.Activation('relu')(xl)
    # xl = layers.Conv2D(32, 3)(xl)
    # xl = layers.BatchNormalization()(xl)
    # xl = layers.Activation('relu')(xl)
    # xl = layers.MaxPooling2D()(xl)
    #
    # xr = layers.Conv2D(32, 3, strides=(2, 2))(inpr)
    # xr = layers.BatchNormalization()(xr)
    # xr = layers.Activation('relu')(xr)
    # xr = layers.Conv2D(32, 3)(xr)
    # xr = layers.BatchNormalization()(xr)
    # xr = layers.Activation('relu')(xr)
    # xr = layers.MaxPooling2D()(xr)

    xc = layers.Conv2D(32, 3, 2)(inpc)
    xc = layers.BatchNormalization()(xc)
    xc = layers.Activation('relu')(xc)
    xc = layers.Conv2D(32, 3)(xc)
    xc = layers.BatchNormalization()(xc)
    xc = layers.Activation('relu')(xc)
    x = layers.MaxPooling2D()(xc)

    # x = layers.concatenate((xl, xc, xr))
    # x = layers.concatenate([xl, xr])

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
    # model = models.Model([inpl, inpr], x)
    model = models.Model(inpc, x)
    model.compile(optimizer='adam',
                  loss='mae')
    return model


def out2d():
    inp = layers.Input((352, 640, 3))
    x = layers.Conv2D(32, 3, strides=(2, 2), padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    d1 = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(d1)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    d2 = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(d2)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    d3 = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(d3)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    d4 = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(d4)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # x = layers.Conv2DTranspose(256, 3, strides=(2, 2), padding='same')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(256, 3, padding='same')(x)
    # # x = layers.concatenate([x, d4])
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2DTranspose(128, 3, strides=(2, 2), padding='same')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(128, 3, padding='same')(x)
    # # x = layers.concatenate([x, d3])
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2DTranspose(64, 3, strides=(2, 2), padding='same')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(64, 3, padding='same')(x)
    # # x = layers.concatenate([x, d2])
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2DTranspose(32, 3, strides=(2, 2), padding='same')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(32, 3, padding='same')(x)
    # # x = layers.concatenate([x, d1])
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(32, 3, padding='same')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
    model = models.Model(inp, x)
    model.compile(optimizer=optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model


def get_data():
    x = []
    y = []
    for file in os.listdir('output/mouse/'):
        if 'jpg' in file:
            continue
        _, l0, l1 = file.split('.')[0].split('_')
        if l0 == '0' and l1 == '0':
            l = 0
        elif l0 == '0' and l1 == '1':
            l = 1
        elif l0 == '1' and l1 == '0':
            l = 2
        elif l0 == '1' and l1 == '1':
            l = 3
        y.append(np.array(l))
        im = cv2.imread(os.path.join('output/mouse/', file[:-4]+'.jpg'))
        ims = im.astype('float32')
        ims /= 255.0
        # ims -= 1
        x.append(ims)
    return np.array(x, dtype='float32'), np.array(y, dtype='int64')


if __name__ == '__main__':
    x, y = get_data()
    x = np.rollaxis(x, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y)

    model = natthaphon.Model(Model())
    # model.summary()
    model.to('cuda')
    model.compile('adam', loss=nn.CrossEntropyLoss(), metric='acc')
    model.fit(x[:-16], y[:-16], epoch=40, batch_size=8,
              # validation_data=[X_test, y_test]
              )
    y_pred = model.predict(x)
    y_pred = sigmoid(y_pred)
    # y_pred = np.einsum('aijk->ajki',y_pred)
    for i in range(y.shape[0]):
        print(y_pred[i], y[i])
        # pred = cv2.resize((y_pred[i] * 255).astype('uint8'), (640, 352))
        # true = cv2.resize((y[i] * 255).astype('uint8'), (640, 352))
        # cv2.imwrite(f'predict/{i}_pred.jpg', pred)
        # cv2.imwrite(f'predict/{i}_test.jpg', true)

