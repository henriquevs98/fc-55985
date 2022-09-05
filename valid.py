'''
Henrique Silva
Faculdade de Ciências da Universidade de Lisboa
Tese Engenharia Geoespacial 2021/2022
'''

import os
import pandas as pd  
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.python.util import deprecation
from unet import modelo_unet 
from sklearn.metrics import recall_score, precision_score

def get_model():
    return modelo_unet(256, 256, 3)
def mask_normalize(mask):
    return mask/(np.amax(mask)+1e-8)
def Average(lst):
    return sum(lst) / len(lst)
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

threshold_intval = 0.05
print('')
print('1. int ') 
print('2. nac')
dir_in = input('Seleccione o número correspondente à pasta de imagens a validar: ')
print('')
print('1. inria ') 
print('2. inria+ ')
print('3. full ') 
print('4. full+ ')
exp = input('Seleccione o número correspondente ao modelo a validar: ')

##########################################_Modelo_##################################################

model = get_model()

if exp == '1':
    model.load_weights('modelos/inria/modelo.hdf5')
    nom = '_inria'
elif exp == '2':
    model.load_weights('modelos/inria+/modelo.hdf5')
    nom = '_inria+'
elif exp == '3':
    model.load_weights('modelos/full/modelo.hdf5')
    nom = '_full'
elif exp == '4':
    model.load_weights('modelos/full+/modelo.hdf5')
    nom = '_full+'

########################################_Leitura_#################################################

print()
print('A ler imagens e respetivas máscaras...')

img = []
predictions = []
trues = []

if dir_in == '1':
    y_true = 'validacao/int/masks/'
    dir_in = 'int'
    
elif dir_in == '2':
    y_true = 'validacao/nac/masks/'
    dir_in = 'nac'

for path, subdirs, files in os.walk(y_true):
    dirname = path.split(os.path.sep)[-1]
    images_est = os.listdir(path)
    for i, image_name in enumerate(images_est):
        img =  cv2.imread('validacao/' + dir_in + '/images/' + image_name, cv2.IMREAD_UNCHANGED)
        img = img[:,:,0:3]
        if exp == '3' or exp == '4':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        y_pred = tf.keras.utils.normalize(np.array(img), axis=1)
        y_pred = np.expand_dims(y_pred, 0)
        y_pred = model.predict(y_pred)[0,:,:,0] 
        predictions.append(y_pred)
        y_true = cv2.imread('validacao/' + dir_in + '/masks/' + image_name, 1)
        y_true = cv2. cvtColor(y_true,cv2. COLOR_BGR2GRAY)
        trues.append(y_true)

#####################################_Validação_##################################################

print()
print('A processar, aguarde...')
print()

f1s = []
recalls = []
precisions = []     

f1s_mean = []
recalls_mean = []
precisions_mean = []
x_graph = []

contagem = threshold_intval 
while round(contagem, 2) != 1:
    for i in range(len(predictions)):
        y_pred = (predictions[i] > contagem).astype(np.uint8)
        y_true = trues[i]
        ######_Recall-score_######
        recall = recall_score(y_true, y_pred, average='micro', zero_division=1)
        recalls.append(recall)
        ######_Precision-score_######
        precision = precision_score(y_true, y_pred, average='micro', zero_division=1)
        precisions.append(precision)
    recalls_mean.append(sum(recalls) / len(recalls))
    precisions_mean.append(sum(precisions) / len(precisions))
    recalls = []
    precisions = []   
    ious = []
    x_graph.append(round(contagem, 2))
    contagem += float(threshold_intval) 
    
    
######_F1-score_######
for i in range(len(recalls_mean)):
    f1 = 2 * (recalls_mean[i] * precisions_mean[i]) / (recalls_mean[i] + precisions_mean[i])
    f1s_mean.append(f1)

plt.plot(x_graph, f1s_mean, label = 'Pontuação F1')
plt.plot(x_graph, recalls_mean, label = 'Recall')
plt.plot(x_graph, precisions_mean, label = 'Precisão')
plt.legend()
plt.show()

print()
print('Número de imagens validadas: ', len(images_est))

###################################_CSV_#####################################################

dictionary = {'Threshold': x_graph,'F1': f1s_mean, 'Recall': recalls_mean, 'Precisao': precisions_mean}  
dataframe = pd.DataFrame(dictionary) 
dir_csv = 'validacao/report/valid' +  nom  + '_' + dir_in + '.csv'
dataframe.to_csv(dir_csv)

text = open(dir_csv, 'r')
text = ''.join([i for i in text]) \
    .replace(',', ';')
text = ''.join([i for i in text]) \
    .replace('.', ',')

x = open(dir_csv,'w')
x.writelines(text)
x.close()

print('')
print('Ficheiro .csv guardado na pasta raíz do programa')

###############################_Analise visual_###############################################

fig = plt.figure(figsize=(10, 7))
rows = 1
columns = 5

fig.add_subplot(rows, columns, 1)
plt.imshow(predictions[17], cmap='gray')
plt.axis('off')
plt.title('original')
  
fig.add_subplot(rows, columns, 2)
plt.imshow((predictions[17] > 0.2).astype(np.uint8), cmap='gray')
plt.axis('off')
plt.title('original > 0.2')
  
fig.add_subplot(rows, columns, 3)
plt.imshow((predictions[17] > 0.4).astype(np.uint8), cmap='gray')
plt.axis('off')
plt.title('original > 0.4')
  
fig.add_subplot(rows, columns, 4)
plt.imshow((predictions[17] > 0.6).astype(np.uint8), cmap='gray' )
plt.axis('off')
plt.title('original > 0.6')

fig.add_subplot(rows, columns, 5)
plt.imshow((predictions[17] > 0.8).astype(np.uint8), cmap='gray')
plt.axis('off')
plt.title('original > 0.8')

predictionsave = (predictions[17] > 0.2) * 255
Image.fromarray(predictionsave).convert('RGB').save(nom + '17int02.png')
predictionsave = (predictions[17] > 0.4) * 255
Image.fromarray(predictionsave).convert('RGB').save(nom + '17int04.png')
predictionsave = (predictions[17] > 0.6) * 255
Image.fromarray(predictionsave).convert('RGB').save(nom + '17int06.png')
predictionsave = (predictions[17] > 0.8) * 255
Image.fromarray(predictionsave).convert('RGB').save(nom + '17int08.png')

print('')
print('Imagens exemplo com diferentes valores de corte guardadas na pasta raíz do programa')