'''
Henrique Silva
Faculdade de Ciências da Universidade de Lisboa
Tese Engenharia Geoespacial 2021/2022
'''

import os
import cv2
import numpy as np
from patchify import patchify
from PIL import Image
import tifffile as tiff

########################################_Configuração_##############################################

img_dir = 'E:/edificios/AIRS/vali'
img_outdir = 'E:/recimgval/'
msk_outdir = 'E:/recmskval/'
patch_size = 256
step = 256

###########################################_Recorte_################################################

est_img = []  
for path, subdirs, files in os.walk(img_dir):
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'images':
        images_est = os.listdir(path)
        for i, image_name in enumerate(images_est):
            image_est = cv2.imread(path+'/'+image_name, cv2.IMREAD_UNCHANGED)
            SIZE_X = (image_est.shape[1]//patch_size)*patch_size
            SIZE_Y = (image_est.shape[0]//patch_size)*patch_size
            image_est = Image.fromarray(image_est)
            image_est = image_est.crop((0 ,0, SIZE_X, SIZE_Y))
            image_est = np.array(image_est)             
   
            print('A recortar imagem:', path+'/'+image_name)
            patches_img_est = patchify(image_est, (patch_size, patch_size, 3), step)
    
            for i in range(patches_img_est.shape[0]):
                for j in range(patches_img_est.shape[1]):
                    
                    single_patch_img_est = patches_img_est[i,j,:,:]
                    
                    single_patch_img_est = (single_patch_img_est.astype('float32')) / 255
                    single_patch_img_est = single_patch_img_est[0]       
                    est_img.append(single_patch_img_est)

for i in range(len(est_img)):
    print('A guardar imagem nº ' + str(i))
    tiff.imwrite(img_outdir + str(i)+ '_AIRS.tif', est_img[i])

est_mask = []
for path, subdirs, files in os.walk(img_dir):
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'masks':
        masks_est = os.listdir(path)
        for i, mask_name in enumerate(masks_est):  
            mask_est = cv2.imread(path+'/'+mask_name, 1)
            mask_est = cv2.cvtColor(mask_est,cv2.COLOR_BGR2RGB)
            SIZE_X = (mask_est.shape[1]//patch_size)*patch_size
            SIZE_Y = (mask_est.shape[0]//patch_size)*patch_size
            mask_est = Image.fromarray(mask_est)
            mask_est = mask_est.crop((0 ,0, SIZE_X, SIZE_Y)) 
            mask_est = np.array(mask_est)             
   
            print('A recortar máscara:', path+'/'+mask_name)
            patches_mask_est = patchify(mask_est, (patch_size, patch_size, 3), step=patch_size)  
    
            for i in range(patches_mask_est.shape[0]):
                for j in range(patches_mask_est.shape[1]):
                    
                    single_patch_mask_est = patches_mask_est[i,j,:,:]
                    single_patch_mask_est = single_patch_mask_est[0]   
                    est_mask.append(single_patch_mask_est)

for i in range(len(est_mask)):    
    print('A guardar máscara nº ' + str(i))       
    tiff.imwrite(msk_outdir + str(i) + '_AIRS.tif', est_mask[i])