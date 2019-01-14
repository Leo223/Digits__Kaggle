"""
Aumentamos el dataset con los digitos que se obtienen del dataset MNIST de Keras

trampa 1.0
"""



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
from keras.datasets import mnist



ruta = os.getcwd()
ruta_train = ruta + '/Data/train.csv'
ruta_test  = ruta + '/Data/test.csv'

data = pd.read_csv(ruta_train)

x_train = data[data.columns[1:]].values
y_train = data[data.columns[0]].values


(x_train2,y_train2),(x_train3,y_train3)=mnist.load_data()
x_train2 = x_train2.reshape(x_train2.shape[0],28,28,1)
x_train3 = x_train3.reshape(x_train3.shape[0],28,28,1)
x_train23 = np.concatenate([x_train2,x_train3])



# Normalizamos las imagenes
x_train = x_train/255.0
# Reestructuramos la estructura de la imagen para la NN
x_train = x_train.reshape(x_train.shape[0],28,28,1)

# Vectorizamos las salidas
y_train  = np_utils.to_categorical(y_train,10)
y_train2 = np_utils.to_categorical(y_train2,10)
y_train3 = np_utils.to_categorical(y_train3,10)

y_train23 = np.concatenate([y_train2,y_train3])


"""
Arquitectura Red Neuronal
"""
model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1)))
model.add(PReLU())
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
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
model.save('Model_newNN_GPU_v10.0.40.h5')

