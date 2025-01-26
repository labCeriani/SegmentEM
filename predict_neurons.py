# https://youtu.be/q-p8v1Bxvac

"""
Author: Dr. Sreenivas Bhattiprolu

Multiclass semantic segmentation using U-Net - prediction on large images
and 3D volumes (slice by slice)

To annotate images and generate labels, you can use APEER (for free):
www.apeer.com 
"""

from keras.utils import normalize
import os
import glob
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
import tensorflow as tf
from models import Attention_ResUNet, UNet, Attention_UNet, dice_coef, dice_coef_loss, jacard_coef
import tifffile

def predict_neurons(volumen,parent_path, project_dir):
    n_classes=3 #Number of classes for segmentation
    IMG_HEIGHT = 256
    IMG_WIDTH  = 256
    IMG_CHANNELS = 1
    patch_size=256

    from focal_loss import BinaryFocalLoss

    #model = UNet((256,256,1), 3)
    model = Attention_ResUNet((256,256,1), 3)
    model.compile(optimizer='adam', loss=BinaryFocalLoss(gamma=2), metrics=['accuracy', jacard_coef])
    #path_weights=os.path.join(parent_path, 'multiclass_UNet_50epochs_B_focal.hdf5') #Dejar esta para final
    path_weights=os.path.join(parent_path, 'multiclass_AttResUnet_50epochs_B_focal.hdf5')
    model.load_weights(path_weights)
    model.summary()

    segm_images = []
    mascara_neuronas_volumen=[]
    mascara_mitocondrias_volumen=[]

    def cut_image(image):
        height, width = image.shape[:2]
        new_height = height - (height % 256)
        new_width = width - (width % 256)
        cut_image = image[:new_height, :new_width]
        return cut_image, height - new_height, width - new_width

    def attach_remaining(image, remaining_height, remaining_width):
        height, width = image.shape[:2]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:height - remaining_height, :width - remaining_width] = image
        return canvas

    from pathlib import Path
    for k in range(volumen.shape[0]):
        large_image=volumen[k,:,:]
        large_image=np.array(large_image)
        large_image_cut, remaining_height, remaining_width = cut_image(large_image)
        patches = patchify(large_image_cut, (256, 256), step=256)  #Step=256 for 256 patches means no overlap
        
        predicted_patches = []
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                
                single_patch = patches[i,j,:,:]
                if single_patch.shape==(256,256):
                    
                    single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
                    single_patch_input=np.expand_dims(single_patch_norm, 0)
            
                    single_patch_prediction = (model.predict(single_patch_input))
                    single_patch_predicted_img=np.argmax(single_patch_prediction, axis=3)[0,:,:]
            
                    predicted_patches.append(single_patch_predicted_img)
        
        predicted_patches = np.array(predicted_patches)
        
        predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 256,256) )
        
        reconstructed_image = unpatchify(predicted_patches_reshaped, large_image_cut.shape)
        reconstructed_whole_image=np.zeros(large_image.shape)
        reconstructed_whole_image[:large_image_cut.shape[0], :large_image_cut.shape[1]] = reconstructed_image
        #aplico prediction al borde de abajo
        patches_borde_abajo = patchify(large_image[large_image.shape[0]-256:,int(remaining_width/2):int(large_image.shape[1]-remaining_width/2)], (256, 256), step=256)  #Step=256 for 256 patches means no overlap
        predicted_patches_borde_abajo = []
        for i in range(patches_borde_abajo.shape[0]):
            for j in range(patches_borde_abajo.shape[1]):
                #print(i,j)
                
                single_patch = patches_borde_abajo[i,j,:,:]
                if single_patch.shape==(256,256):
                    
                    single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
                    single_patch_input=np.expand_dims(single_patch_norm, 0)
            
                    single_patch_prediction = (model.predict(single_patch_input))
                    single_patch_predicted_img=np.argmax(single_patch_prediction, axis=3)[0,:,:]
            
                    predicted_patches_borde_abajo.append(single_patch_predicted_img)

        predicted_patches_borde_abajo = np.array(predicted_patches_borde_abajo)

        predicted_patches_borde_abajo_reshaped = np.reshape(predicted_patches_borde_abajo, (patches_borde_abajo.shape[0], patches_borde_abajo.shape[1], 256,256) )

        reconstructed_image_borde_abajo = unpatchify(predicted_patches_borde_abajo_reshaped, (256,large_image_cut.shape[1]))
        reconstructed_whole_image[large_image.shape[0]-256:, int(remaining_width/2):int(large_image.shape[1]-remaining_width/2)] = reconstructed_image_borde_abajo

        #aplico prediction al borde lateral
        patches_borde_lateral = patchify(large_image[int(remaining_height/2):int(large_image.shape[0]-remaining_height/2),large_image.shape[1]-256:], (256, 256), step=256)  #Step=256 for 256 patches means no overlap
        predicted_patches_borde_lateral = []
        for i in range(patches_borde_lateral.shape[0]):
            for j in range(patches_borde_lateral.shape[1]):
                #print(i,j)
                
                single_patch = patches_borde_lateral[i,j,:,:]
                if single_patch.shape==(256,256):
                    
                    single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
                    single_patch_input=np.expand_dims(single_patch_norm, 0)
            
                    single_patch_prediction = (model.predict(single_patch_input))
                    single_patch_predicted_img=np.argmax(single_patch_prediction, axis=3)[0,:,:]
            
                    predicted_patches_borde_lateral.append(single_patch_predicted_img)

        predicted_patches_borde_lateral = np.array(predicted_patches_borde_lateral)

        predicted_patches_borde_lateral_reshaped = np.reshape(predicted_patches_borde_lateral, (patches_borde_lateral.shape[0], patches_borde_lateral.shape[1], 256,256) )

        reconstructed_image_borde_lateral = unpatchify(predicted_patches_borde_lateral_reshaped, (large_image_cut.shape[0],256))
        reconstructed_whole_image[int(remaining_height/2):int(large_image.shape[0]-remaining_height/2),large_image.shape[1]-256:] = reconstructed_image_borde_lateral
        #GENERO UNA MASCARA BLANCA PARA MITOCONDRIAS Y NEURONAS, 1 neurona 2 mitocondria
        mascara=reconstructed_whole_image.copy()
        mascara = mascara.astype(np.uint8)
        mascara[reconstructed_whole_image == 1] = 255  # Cambiar los píxeles con valor 1 a 255
        mascara[reconstructed_whole_image == 2] = 255  # Cambiar los píxeles con valor 2 a 255
        #EROSION Y DILATACION
        kernel_morf = np.ones((5,5), np.uint8)
        eroded_image = cv2.erode(mascara, kernel_morf, iterations=1)
        mascara= cv2.dilate(eroded_image, kernel_morf, iterations=1)
        #FILTRADO POR NRO DE PIXELS AGRUPADOS
        min_pixels_por_aglomeracion=300
        # Encontrar las regiones conectadas en la imagen binarizada
        num_regiones, etiquetas = cv2.connectedComponents(mascara)
        filtrada=mascara.copy()
        # Calcular el tamaño de cada región
        tamanos = np.bincount(etiquetas.flatten())
        # Eliminar las regiones que tienen menos píxeles que el umbral especificado
        for etiqueta, tamano in enumerate(tamanos):
            if tamano > 0 and tamano < min_pixels_por_aglomeracion:
                filtrada[etiquetas == etiqueta] = 0  # Establecer píxeles de la región a negro
        mascara_neuronas=np.zeros_like(reconstructed_whole_image)
        mascara_neuronas[(filtrada>=200) & (reconstructed_whole_image>=1)]=255
        mascara_mitocondrias=np.zeros_like(reconstructed_whole_image)
        mascara_mitocondrias[(filtrada>=200) & (reconstructed_whole_image==1)]=255
        mascara_neuronas_volumen.append(mascara_neuronas)
        mascara_mitocondrias_volumen.append(mascara_mitocondrias)
        #reconstructed_whole_image[filtrada == 0] = 0
        #segm_images.append(reconstructed_whole_image)
        print("Finished prediciting image: ", k)
        
        
    final_segm_image_mitocondrias = np.array(mascara_mitocondrias_volumen).astype(np.uint8)
    final_segm_image_neuronas = np.array(mascara_neuronas_volumen).astype(np.uint8)

    path_neurons= os.path.join(project_dir, 'mascara_neuronas.tif')
    path_mitocondria=os.path.join(project_dir, 'mascara_mitocondrias.tif')
    from tifffile import imsave
    imsave(path_neurons, np.transpose(final_segm_image_neuronas, axes=(0, 2, 1)))
    imsave(path_mitocondria, np.transpose(final_segm_image_mitocondrias, axes=(0, 2, 1)))
    #imsave(path_neurons, final_segm_image_neuronas)
    #imsave(path_mitocondria,final_segm_image_mitocondrias)
    return final_segm_image_neuronas, final_segm_image_mitocondrias
   
    
    
#parent_path=r'\Users\ulabceriani\Documents\ITBA'
#volumen=tifffile.imread(r'\Users\ulabceriani\Documents\ITBA\pruebainterfaz\prueba_zt22\volumen.tif')
#project_dir=r'\Users\ulabceriani\Documents\ITBA\resultados_multiclass_unet'
#print(volumen.shape)
#segmentacion_neuronas, segmentacion_mitocondrias=predict_neurons(volumen, parent_path, project_dir)

    
    
    
    
    
    
    