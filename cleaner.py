"""
Henrique Silva
Faculdade de Ciências da Universidade de Lisboa
Tese Engenharia Geoespacial 2021/2022
"""


import os
import numpy as np
import cv2
import tifffile as tiff


def remocao_maskbin0(extensao, caminho): #remover máscaras binárias que não contêm 1's
    mask_array = []
    imagens_dir = os.listdir(caminho)
    for i, image_name in enumerate(imagens_dir):    
        if (image_name.split('.')[1] == extensao):
            image = cv2.imread(caminho + image_name, cv2.IMREAD_UNCHANGED)
            mask_array.append(np.array(image))
    for c in range(len(mask_array)):
        if np.sum(mask_array[c])==0:
            os.remove(caminho + str(imagens_dir[c]))
    print('')
    print('Processo de remoção de máscaras binárias neutras concluído.')
    
def comparacao_remocao_img(extensao, mask_path, img_path, pathtosave): #remover imagens que não existem na pasta mask e guardar no path
    img = os.listdir(img_path)
    mask = os.listdir(mask_path)
    print('')
    print('A compatibilizar imagens...' )
    cont = 0
    while len(img) != len(mask):
        for filename in img:
            if filename not in mask:
                cont += 1
                print (cont)
                img.remove(filename)
    array = []
    for i, image_name in enumerate(img):    
        if (image_name.split('.')[1] == extensao):
            print('Associação com ' + str(len(array) + 1) + ' imagens')
            image = cv2.imread(img_path + image_name, cv2.IMREAD_UNCHANGED)
            array.append(np.array(image))
    for i in range(len(array)):
        print(str(i))
        tiff.imwrite(pathtosave + str(img[i]), array[i])
                

remocao_maskbin0('tif', 
                  'E:/recmsk/')

comparacao_remocao_img('tif', 'E:/recmsk', 
                        'E:/edificios/AIRS/recimg/',
                        'E:/edificios/recimg/')


