import pandas as pd
import os

from keras.models import load_model

ruta = os.getcwd()
ruta_test = ruta + '/Data/test.csv'

x_test_original = pd.read_csv(ruta_test).values


model = load_model('Model_newNN_GPU_v5.5.200.h5')

def load_data(im):
    # Normalizamos las imagenes
    im = im/255.00
    # Reestructuramos la estructura de la imagen para la NN
    im = im.reshape(1, 28, 28, 1)
    return im


def Predict(ima_test, model=model):
    x_test = load_data(ima_test)
    y_pred = model.predict(x_test)
    prediccion = list(y_pred[0]).index(y_pred[0].max())
    return prediccion


pred_dict = {}
size = float(len(x_test_original))
for indice, imagen in enumerate(x_test_original):
    pred_dict[indice+1] = Predict(imagen)
    print(str(round((float(indice)/size)*100, 2)) + '%')

df1 = pd.DataFrame({'ImageId': list(pred_dict.keys()), 'Label': list(pred_dict.values())}).set_index('ImageId')

df1.to_csv('./Data/Digits_out_v5.5.200.csv')





