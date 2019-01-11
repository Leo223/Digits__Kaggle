"""
En este procedimiento de inferencia se llama a varias estructuras de Red neuronal que clasifican

- los numeros naturales (0,1,2,3,4,5,6,7,8,9) --> 'Model_newNN_GPU_v5.5.40.h5'
- Grano fino: Diferencia entre el 4 y 9       --> 'Model_newNN_GPU_4&9_v7.0.h5'

"""
import pandas as pd
import numpy as np
import os

from keras.models import load_model

ruta = os.getcwd()
ruta_test = ruta + '/Data/test.csv'

x_test_original = pd.read_csv(ruta_test).values

dic_num49={0:4,1:9}
dic_num38={0:3,1:8}

model   = load_model('Model_newNN_GPU_v5.5.40.h5')
model49 = load_model('Model_newNN_GPU_4&9_v7.0.h5')
model38 = load_model('Model_newNN_GPU_3&8_v7.0.h5')

def load_data(im):
    # Normalizamos las imagenes
    im = im/255.00
    # Reestructuramos la estructura de la imagen para la NN
    im = im.reshape(1, 28, 28, 1)
    return im

def func_49(x_test):
    y_pred = model49.predict(x_test)
    prediccion = dic_num49.get(np.argmax(y_pred))
    return prediccion

def func_38(x_test):
    y_pred = model38.predict(x_test)
    prediccion = dic_num38.get(np.argmax(y_pred))
    return prediccion


def Predict(ima_test, model=model,model49=model49):
    x_test = load_data(ima_test)
    y_pred = model.predict(x_test)
    prediccion = np.argmax(y_pred[0])

    if prediccion in [4,9]:
        prediccion = func_49(x_test)

    if prediccion in [3,8]:
        prediccion = func_38(x_test)

    return prediccion


pred_dict = {}
size = float(len(x_test_original))
for indice, imagen in enumerate(x_test_original):
    pred_dict[indice+1] = Predict(imagen)
    print(str(round((float(indice)/size)*100, 2)) + '%')

df1 = pd.DataFrame({'ImageId': list(pred_dict.keys()), 'Label': list(pred_dict.values())}).set_index('ImageId')

df1.to_csv('./Data/Digits_out_v8.0.csv')

## v
## v7.0 (4&9)     -->   Kaggle: Pos: --- acc: 0.99685
## v7.0 (4&9 3&8) -->   Kaggle: Pos: --- acc: 0.99671

