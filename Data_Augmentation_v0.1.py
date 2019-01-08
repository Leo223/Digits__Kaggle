import pandas as pd
import numpy as np
from PIL import Image,ImageFilter

import os
from keras.preprocessing.image import ImageDataGenerator


ruta = os.getcwd()
ruta_train = ruta + '/Data/train.csv'
ruta_test = ruta + '/Data/test.csv'

data = pd.read_csv(ruta_train)

# Obtenemos solo las filas de los 4 y los 9
data_filter = data [(data['label'] == 4) | (data['label'] == 9)]






x_train = data[data.columns[1:]].values
y_train = data[data.columns[0]].values


# Normalizamos las imagenes
# x_train = x_train/255.0
# Reestructuramos la estructura de la imagen para la NN
# x_train = x_train.reshape(x_train.shape[0],28,28,1)




# Data Augmentation
# datagen = ImageDataGenerator(
#     rotation_range=10,
#     zoom_range=0.1,
#     width_shift_range=0.1,
#     height_shift_range=0.1)
# datagen.fit(x_train)


