'''
Henrique Silva
Faculdade de Ciências da Universidade de Lisboa
Tese Engenharia Geoespacial 2021/2022
'''

from unet import modelo_unet 
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.util import deprecation
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import  recall_score, precision_score

def get_model():
    return modelo_unet(256, 256, 3)
def tensorflow_shutup():
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        def deprecated(date, instructions, warn_once=True):  
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper
        deprecation.deprecated = deprecated
    except ImportError:
        pass
tensorflow_shutup()

########################################_Configuração_##############################################

img_in = 'validacao/val256/images/37_Orto.tif'
msk_in = 'validacao/val256/masks/37_Orto.tif'
print('')
print('1. inria ')
print('2. inria+ ')
print('3. full ')
print('4. full+ ')
exp = input('Seleccione o número correspondente ao modelo a validar: ')
threshold = float(input('Indique o valor de corte: '))

##########################################_Modelo_##################################################

model = get_model()

if exp == '1':
    model.load_weights('modelos/inria/modelo.hdf5')
elif exp == '2':
    model.load_weights('modelos/inria+/modelo.hdf5')
elif exp == '3':
    model.load_weights('modelos/full/modelo.hdf5')
elif exp == '4':
    model.load_weights('modelos/full+/modelo.hdf5')
    
########################################_Validação_##################################################

img =  cv2.imread(img_in, cv2.IMREAD_UNCHANGED)
img = img[:,:,0:3]

y_pred = tf.keras.utils.normalize(np.array(img), axis=1)
y_pred = np.expand_dims(y_pred, 0)
y_pred = (model.predict(y_pred)[0,:,:,0] > threshold).astype(np.uint8) * 255
y_true = cv2.imread(msk_in, 1)
y_true = cv2. cvtColor(y_true,cv2. COLOR_BGR2GRAY)

fig = plt.figure(figsize=(10, 10))
rows = 1
columns = 3
fig.add_subplot(rows, columns, 1)
plt.imshow(img)
plt.axis('off')
plt.title('Imagem')
fig.add_subplot(rows, columns, 2)
plt.imshow(y_pred)
plt.axis('off')
plt.title('Previsão')
fig.add_subplot(rows, columns, 3)
plt.imshow(y_true)
plt.axis('off')
plt.title('Verdadeiro')

######_Recall-score_######
recall = recall_score(y_true, y_pred, average='micro')
print('Recall: ', recall)
######_Precision-score_######
precision = precision_score(y_true, y_pred, average='micro', zero_division=1)
print('Precisão: ', precision)
######_F1-score_######
f1 = 2 * (recall * precision) / (recall + precision)
print('Pontuação F1: ', f1)

Image.fromarray(y_pred).convert('RGB').save('validprev256.png')
