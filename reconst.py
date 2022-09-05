"""
Henrique Silva
Faculdade de Ciências da Universidade de Lisboa
Tese Engenharia Geoespacial 2021/2022
"""

import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import numpy as np
from patchify import patchify
from tensorflow.python.util import deprecation
from unet import modelo_unet 

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

###############################################_Configuração_#############################################

img_dir = 'D:/Ambiente de Trabalho/concatbin/img'
patch_size = 256
step = 128
threshold = float(input('Indique o valor de corte: '))

#################################################_Modelo_###############################################

model = get_model()

model.load_weights('modelos/full+/modelo.hdf5')

###########################################_Recorte_################################################

print('')
print('A recortar imagens..')

rec_img = []  
prev = []

for path, subdirs, files in os.walk(img_dir):
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'images':
        images = os.listdir(path)
        for i, image_name in enumerate(images[:1]):
            imgorg = cv2.imread(path+'/'+image_name, cv2.IMREAD_UNCHANGED)
            SIZE_X = (imgorg.shape[1]//patch_size)*patch_size
            SIZE_Y = (imgorg.shape[0]//patch_size)*patch_size
            image = Image.fromarray(imgorg)
            image = image.crop((0 ,0, SIZE_X, SIZE_Y))
            image = np.array(image)             
            patches_img = patchify(image, (patch_size, patch_size, 3), step)
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i,j,:,:]
                    single_patch_img = (single_patch_img.astype('float32')) / 255
                    single_patch_img = single_patch_img[0]       
                    rec_img.append(single_patch_img)

###########################################_Previsão_################################################

print('')
print('A prever em imagens recortadas...')

prevf = [] 

for i in range(len(rec_img)):
    prev = tf.keras.utils.normalize(np.array(rec_img[i]), axis=1)
    prev = np.expand_dims(prev, 0)
    prev = model.predict(prev)[0,:,:,0] 
    prev = (prev > threshold).astype(np.uint8)
    prevf.append(prev)

###########################################_Reconstrução_################################################

print('')
print('A reconstruir imagem...')

lin = int(image.shape[0] / step)- 1
col = int(image.shape[1] / step) - 1
c = 0
mskh = []

for j in range(lin):
    msk = prevf[0 + col*c]
    for i in range(col - 1):
        msk2 = cv2.copyMakeBorder(prevf[i + 1 + col*c], 0, 0, step * (i + 1), 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        msk = cv2.copyMakeBorder(msk, 0, 0, 0, step, cv2.BORDER_CONSTANT, value=[0, 0, 0]) + msk2
    c += 1
    msk[msk > 1] = 1
    mskh.append(msk)

mski = mskh[0]
for i in range(lin - 1):
    msk2 = cv2.copyMakeBorder(mskh[i + 1], step * (i + 1), 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]) 
    mski = cv2.copyMakeBorder(mski, 0, step, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]) + msk2
mski[mski > 1] = 1


mski = cv2.copyMakeBorder(mski, 0,imgorg.shape[0] - mski.shape[0] , 0, imgorg.shape[1] - mski.shape[1], 
                          cv2.BORDER_CONSTANT, value=[0, 0, 0]) 

Image.fromarray(mski * 255).convert('RGB').save('teste.png')
plt.imshow((mski) , cmap='gray')

print()
print('Imagem de previsão reconstruída guardada na pasta raíz do programa')