#!/usr/bin/env python2
"""
Author: YongShan
Date: 2017/05/06
"""
import csv
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.models import load_model
from keras.utils import np_utils
from itertools import islice
from os.path import join, isfile
from datetime import datetime
from os import listdir
import sys
import face_detect

DATA_DIR = '.'
X_TRAIN_NPY = 'x_train.npy'
Y_TRAIN_NPY = 'y_train.npy'
X_VAL_NPY = 'x_val.npy'
Y_VAL_NPY = 'y_val.npy'

CATEGORY_EMOTION = 7
LEN_PIXEL = 48
LEN_VAL = 28709
LEN_TEST = 7178
x_train = np.zeros((LEN_VAL, LEN_PIXEL, LEN_PIXEL, 1))
y_train = np.zeros((LEN_VAL, 1))
x_val = np.zeros((LEN_TEST, LEN_PIXEL, LEN_PIXEL, 1))
y_val = np.zeros((LEN_TEST, 1))
model = Sequential()

def csv2npy():
    with open('fer2013.csv') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for index, line in enumerate(islice(csv_reader, 1, None)):
            if index < LEN_VAL:
                x_train[index] = np.asarray(map(int, line[1].split(' '))).reshape((LEN_PIXEL, LEN_PIXEL, 1))
                y_train[index] = int(line[0])
            else:
                x_val[index - LEN_VAL] = np.asarray(map(int, line[1].split(' '))).reshape((LEN_PIXEL, LEN_PIXEL, 1))
                y_val[index - LEN_VAL] = int(line[0])
        np.save(join(DATA_DIR, X_TRAIN_NPY), x_train)
        np.save(join(DATA_DIR, Y_TRAIN_NPY), y_train)
        np.save(join(DATA_DIR, X_VAL_NPY), x_val)
        np.save(join(DATA_DIR, Y_VAL_NPY), y_val)

def load_npy():
    print '[+]Loading data'
    global x_train
    x_train = np.load(join(DATA_DIR, X_TRAIN_NPY)).astype('float32')
    global y_train
    y_train = np_utils.to_categorical(np.load(join(DATA_DIR, Y_TRAIN_NPY)).astype('float32'), CATEGORY_EMOTION)
    global x_val
    x_val = np.load(join(DATA_DIR, X_VAL_NPY)).astype('float32')
    global y_val
    y_val = np_utils.to_categorical(np.load(join(DATA_DIR, Y_VAL_NPY)).astype('float32'), CATEGORY_EMOTION)

def trainCNN():
    global model, x_train, y_train, x_val, y_val
    print '[+]Building CNN'

    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(LEN_PIXEL, LEN_PIXEL, 1)))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2)))
    model.add(keras.layers.pooling.MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))

    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.pooling.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1)))
    model.add(keras.layers.pooling.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(CATEGORY_EMOTION, activation='softmax'))

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    print '[+]Training CNN'
    history_callback = model.fit(x_train, y_train, batch_size=128, epochs=32, validation_data=(x_val, y_val))
    loss_history = history_callback.history["loss"]
    numpy_loss_history = np.array(loss_history)
    now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    np.savetxt(now + "train_loss_history.txt", numpy_loss_history, delimiter=",")
    model.save('cnn.hdf5')

def printUsage():
    print 'Usage: python2 main.py [arg]'
    print '                       generate: generate data'
    print '                       train: train model'
    print '                       test: test model'

def testCNN():
    global model
    model = load_model('cnn.hdf5')
    files = [f for f in listdir('testset') if isfile(join('testset', f))]
    for 
    x_test = np.zeros((LEN_VAL, LEN_PIXEL, LEN_PIXEL, 1))
    y_test = np.zeros((LEN_VAL, 1))
    score = model.evaluate(x_val, y_val, batch_size=128)
    print '[+]Test set: '
    print '   Test score:', score[0]
    print '   Test accuracy:', score[1]
    print '[+]Done'

if __name__ == '__main__':
    if len(sys.argv) != 2:
        printUsage()
        exit(1)
    arg = sys.argv[1]
    if arg == 'generate':
        csv2npy()
        exit(0)
    elif arg == 'train':
        load_npy()
        trainCNN()
        exit(0)
    elif arg == 'test':
        load_npy()
        testCNN()
        exit(0)
    else:
        printUsage()
        exit(0)

