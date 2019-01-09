"""
Competicion de Kaggle (Digits)

Reconocimiento de digitos escritos a mano:
    - Procesamos imagenes de (28x28x1)
    - Red Neuronal creada desde cero manualmente:
        * Input (28x28x1)
        * Conv2D (64f, kernel=5x5)
        * Conv2D (64f, kernel=5x5)
        * Conv2D (64f, kernel=3x3)
        * Pooling (Max, 2x2)
        * Dropout (0.25)
        ------
        * Conv2D (32f, kernel=3x3)
        * Conv2D (32f, kernel=3x3)
        * Pooling (Max, 2x2)
        * Dropout (0.25)
        -----
        * Conv2D (32f, kernel=3x3)
        * Conv2D (32f, kernel=3x3)
        * Pooling (Max, 2x2)
        * Dropout (0.25)
        ------
        * Full Connected (Flatten)
        * Capa 256 neuronas
        * Dropout (0.25)
        * Output (10 clases)

    - Data Augmentation:
        * rotation_range = 10, Rango de rotacion en 10º
        * zoom_range = 0.1, 10% de zoom de la imagen
        * width_shift_range = 0.1, 10% de movimiento lateral de la imagen
        * height_shift_range = 0.1, 10% de movimiento vertical de la imagen

Modelo:
-AWS (GPU=V100)
acc_model:  loss: 0.0597 - acc: 0.9837
-Local
acc_model: loss:  - acc:


Resultado Leaderboard Kaggle:
acc: 0.99300
Pos: 818

"""

import pandas as pd
import numpy as np
from PIL import Image,ImageFilter

import os

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model



ruta = os.getcwd()
ruta_train = ruta + '/Data/train.csv'
ruta_test = ruta + '/Data/test.csv'

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
model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

# model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
#                  activation ='relu'))
# model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
#                  activation ='relu'))
# model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model.add(Dropout(0.25))

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


epochs = 30 #
batch_size = 86
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs,
                              verbose = 1,
                              # steps_per_epoch = x_train.shape[0] // batch_size,
                              steps_per_epoch = 1000,
                              callbacks = [learning_rate_reduction])

# Persistimos el modelo
model.save('Model_newNN_GPU_v4.3.h5')



