import pandas as pd
import numpy as np
from PIL import Image,ImageFilter

import os

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model
from keras.layers.advanced_activations import PReLU



ruta = os.getcwd()
ruta_train = ruta + '/train.csv'
ruta_test  = ruta + '/test.csv'

data = pd.read_csv(ruta_train)

x_train = data[data.columns[1:]].values
y_train = data[data.columns[0]].values

# Normalizamos las imagenes
x_train = x_train/255.0
# Reestructuramos la estructura de la imagen para la NN
x_train = x_train.reshape(x_train.shape[0],28,28,1)

# Vectorizamos las salidas
y_train = np_utils.to_categorical(y_train,10)



"""
Arquitectura Red Neuronal
"""
model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1)))
model.add(PReLU())
model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(PReLU())
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(PReLU())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))


model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(PReLU())
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(PReLU())
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(PReLU())
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(PReLU())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# funcion para modificar el factor de aprendizaje en funcion de su evolucion
learning_rate_reduction = ReduceLROnPlateau(monitor='acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1)
datagen.fit(x_train)


epochs = 50 #
batch_size = 86
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs,
                              verbose = 1,
                              # steps_per_epoch = x_train.shape[0] // batch_size,
                              steps_per_epoch = 1000,
                              callbacks = [learning_rate_reduction])

# Persistimos el modelo
model.save('Model_newNN_GPU_v5.2.40.h5')

##  50 epochs/activation      relu: Model --> loss: 0.0148 - acc: 0.9954 // Kaggle: Pos: 203 acc: 0.99700
##  50 epochs/activation    linear: Model --> loss: 0.0087 - acc: 0.9976 // Kaggle: Pos: 183 acc: 0.99714 !!
##  50 epochs/activation Leakyrelu: Model --> loss: 0.0114 - acc: 0.9970 // Kaggle: Pos: --- acc: 0.99528

##  60e (3,3)/activation lin/PRelu: Model --> loss: 0.0082 - acc: 0.9975 // Kaggle: Pos: 158 acc: 0.99742
