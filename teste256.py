'''
Henrique Silva
Faculdade de Ciências da Universidade de Lisboa
Tese Engenharia Geoespacial 2021/2022
'''

import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from unet import modelo_unet 
from tensorflow.python.util import deprecation
from PIL import Image

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

img_in = 'D:/images/1607.tif'

##########################################_Modelo_#################################################
model = get_model()

model.load_weights('modelos/full+/modelo.hdf5')

##########################################_Teste_##################################################

test_img = cv2.imread(img_in, cv2.IMREAD_UNCHANGED)
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
# test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img_norm = tf.keras.utils.normalize(np.array(test_img), axis=1)
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.3).astype(np.uint8)

######################################_Analise visual_############################################

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Imagem Externa')
plt.imshow(test_img, cmap='gray')
plt.subplot(232)
plt.title('Previsão em Imagem Externa')
plt.imshow(prediction, cmap='gray')
plt.show()

Image.fromarray(prediction).convert('RGB').save('testeprev256.png')