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


ruta = os.getcwd()
ruta_train = ruta + '/Data/train.csv'
ruta_test = ruta + '/Data/test.csv'

data = pd.read_csv(ruta_train)

# Obtenemos solo las filas de los 4 y los 9
data_filter = data [(data['label'] == 4) | (data['label'] == 9)]

# 0 => 4
# 1 => 9
data_num = data_filter.label.values
for ind,_val in enumerate(data_num):
    if _val == 4:
        data_num[ind] = 0
    else:
        data_num[ind] = 1
data_filter = data_filter[data_filter.columns[1:]]
data_filter['label'] = data_num

x_train = data_filter[data_filter.columns[:-1]].values
y_train = data_filter[data_filter.columns[-1]].values

# Normalizamos las imagenes
x_train = x_train/255.0
# Reestructuramos la estructura de la imagen para la NN
x_train = x_train.reshape(x_train.shape[0],28,28,1)

# Vectorizamos las salidas
y_train = np_utils.to_categorical(y_train,2)



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
# model.add(Dense(10, activation = "softmax"))
model.add(Dense(2, activation='sigmoid'))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"])


# funcion para modificar el factor de aprendizaje en funcion de su evolucion
learning_rate_reduction = ReduceLROnPlateau(monitor='acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1)
datagen.fit(x_train)

epochs = 30 #
batch_size = 86
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs,
                              verbose = 1,
                              # steps_per_epoch = x_train.shape[0] // batch_size,
                              steps_per_epoch = 100,
                              callbacks = [learning_rate_reduction])

# Persistimos el modelo
model.save('Model_newNN_GPU_4&9_v7.0.h5')


# Epoch 60/60
# 1000/1000 [==============================] - 24s 24ms/step - loss: 8.3809e-04 - acc: 0.9999