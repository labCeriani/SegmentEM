from simple_unet_model import simple_unet_model 

from keras.utils import normalize
import os
import glob
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
import tifffile
from models import Attention_ResUNet_formito, UNet, Attention_UNet, dice_coef, dice_coef_loss, jacard_coef


def multiply_mask(tiff,mascara,path,guardar):
    mascara_binaria = (mascara >= 250).astype(np.uint8)
    multiplied = tiff * mascara_binaria
    if guardar:
        path_multiplied= os.path.join(path, 'multiplied.tif')
        tifffile.imwrite(path_multiplied, np.transpose(multiplied, axes=(0, 2, 1)))
    return multiplied

def predict_mito(volumen, parent_path, project_dir):
    n_classes=1 #Number of classes for segmentation
    IMG_HEIGHT = 256
    IMG_WIDTH  = 256
    IMG_CHANNELS = 1
    patch_size=256

    def get_model():
        return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    model = get_model()
    path_weights=os.path.join(parent_path, 'simple_multiplied_unet_epoch50.hdf5')
    model.load_weights(path_weights)

    segm_images = []

    def cut_image(image):
        # Get the dimensions of the image
        height, width = image.shape[:2]
        # Calculate the new dimensions that are divisible by 256
        new_height = height - (height % 256)
        new_width = width - (width % 256)
        # Cut the image to the new dimensions
        cut_image = image[:new_height, :new_width]
        return cut_image, height - new_height, width - new_width

    def attach_remaining(image, remaining_height, remaining_width):
        # Get the dimensions of the image
        height, width = image.shape[:2]
        # Create an empty canvas with dimensions equal to the original image
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # Place the processed image onto the canvas
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
                    single_patch_shape = single_patch_norm.shape[:2]
                    single_patch_norm=single_patch_norm[:,:,0][:,:,None]
                    single_patch_input=np.expand_dims(single_patch_norm, 0)
                    single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.2).astype(np.uint8)      
                    predicted_patches.append(cv2.resize(single_patch_prediction, single_patch_shape[::-1]))
            
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
                
                single_patch = patches_borde_abajo[i,j,:,:]
                if single_patch.shape==(256,256):
                    
                    single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
                    single_patch_shape = single_patch_norm.shape[:2]
                    single_patch_norm=single_patch_norm[:,:,0][:,:,None]
                    single_patch_input=np.expand_dims(single_patch_norm, 0)
                    single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.2).astype(np.uint8)        
                    predicted_patches_borde_abajo.append(cv2.resize(single_patch_prediction, single_patch_shape[::-1]))

        predicted_patches_borde_abajo = np.array(predicted_patches_borde_abajo)

        predicted_patches_borde_abajo_reshaped = np.reshape(predicted_patches_borde_abajo, (patches_borde_abajo.shape[0], patches_borde_abajo.shape[1], 256,256) )

        reconstructed_image_borde_abajo = unpatchify(predicted_patches_borde_abajo_reshaped, (256,large_image_cut.shape[1]))
        reconstructed_whole_image[large_image.shape[0]-256:, int(remaining_width/2):int(large_image.shape[1]-remaining_width/2)] = reconstructed_image_borde_abajo

        #aplico prediction al borde lateral
        patches_borde_lateral = patchify(large_image[int(remaining_height/2):int(large_image.shape[0]-remaining_height/2),large_image.shape[1]-256:], (256, 256), step=256)  #Step=256 for 256 patches means no overlap
        predicted_patches_borde_lateral = []
        for i in range(patches_borde_lateral.shape[0]):
            for j in range(patches_borde_lateral.shape[1]):
                
                single_patch = patches_borde_lateral[i,j,:,:]
                if single_patch.shape==(256,256):
                    
                    single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
                    single_patch_shape = single_patch_norm.shape[:2]
                    single_patch_norm=single_patch_norm[:,:,0][:,:,None]
                    single_patch_input=np.expand_dims(single_patch_norm, 0)
                    single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.2).astype(np.uint8)      
                    predicted_patches_borde_lateral.append(cv2.resize(single_patch_prediction, single_patch_shape[::-1]))

        predicted_patches_borde_lateral = np.array(predicted_patches_borde_lateral)

        predicted_patches_borde_lateral_reshaped = np.reshape(predicted_patches_borde_lateral, (patches_borde_lateral.shape[0], patches_borde_lateral.shape[1], 256,256) )

        reconstructed_image_borde_lateral = unpatchify(predicted_patches_borde_lateral_reshaped, (large_image_cut.shape[0],256))
        reconstructed_whole_image[int(remaining_height/2):int(large_image.shape[0]-remaining_height/2),large_image.shape[1]-256:] = reconstructed_image_borde_lateral
        reconstructed_whole_image[reconstructed_whole_image == 1] = 255  # Cambiar los píxeles con valor 1 a 255
        kernel_morf = np.ones((5,5), np.uint8)
        eroded_image = cv2.erode(reconstructed_whole_image, kernel_morf, iterations=1)
        reconstructed_whole_image = cv2.dilate(eroded_image, kernel_morf, iterations=1)
        segm_images.append(reconstructed_whole_image)
        
        
    final_segm_image = np.array(segm_images).astype(np.uint8)   
    path_mitocondria= os.path.join(project_dir, 'mascara_mitocondrias.tif')
    tifffile.imsave(path_mitocondria, np.transpose(final_segm_image, axes=(0, 2, 1)))
    return final_segm_image

def predict_mito_resattunet(volumen, parent_path, project_dir):
    n_classes=1 #Number of classes for segmentation
    IMG_HEIGHT = 256
    IMG_WIDTH  = 256
    IMG_CHANNELS = 1
    patch_size=256

    from focal_loss import BinaryFocalLoss

    #model = UNet((256,256,1), 3)
    model = Attention_ResUNet_formito((256,256,1), 1)
    model.compile(optimizer='adam', loss=BinaryFocalLoss(gamma=2), metrics=['accuracy', jacard_coef])
    #path_weights=os.path.join(parent_path, 'multiclass_UNet_50epochs_B_focal.hdf5') #Dejar esta para final
    path_weights=os.path.join(parent_path, 'simple_class_AttResUnet_50epochs_B_focal_2.hdf5')
    model.load_weights(path_weights)
    model.summary()

    segm_images = []

    def cut_image(image):
        # Get the dimensions of the image
        height, width = image.shape[:2]
        # Calculate the new dimensions that are divisible by 256
        new_height = height - (height % 256)
        new_width = width - (width % 256)
        # Cut the image to the new dimensions
        cut_image = image[:new_height, :new_width]
        return cut_image, height - new_height, width - new_width

    def attach_remaining(image, remaining_height, remaining_width):
        # Get the dimensions of the image
        height, width = image.shape[:2]
        # Create an empty canvas with dimensions equal to the original image
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # Place the processed image onto the canvas
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
                    single_patch_shape = single_patch_norm.shape[:2]
                    single_patch_norm=single_patch_norm[:,:,0][:,:,None]
                    single_patch_input=np.expand_dims(single_patch_norm, 0)
                    single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.2).astype(np.uint8)      
                    predicted_patches.append(cv2.resize(single_patch_prediction, single_patch_shape[::-1]))
            
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
                
                single_patch = patches_borde_abajo[i,j,:,:]
                if single_patch.shape==(256,256):
                    
                    single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
                    single_patch_shape = single_patch_norm.shape[:2]
                    single_patch_norm=single_patch_norm[:,:,0][:,:,None]
                    single_patch_input=np.expand_dims(single_patch_norm, 0)
                    single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.2).astype(np.uint8)        
                    predicted_patches_borde_abajo.append(cv2.resize(single_patch_prediction, single_patch_shape[::-1]))

        predicted_patches_borde_abajo = np.array(predicted_patches_borde_abajo)

        predicted_patches_borde_abajo_reshaped = np.reshape(predicted_patches_borde_abajo, (patches_borde_abajo.shape[0], patches_borde_abajo.shape[1], 256,256) )

        reconstructed_image_borde_abajo = unpatchify(predicted_patches_borde_abajo_reshaped, (256,large_image_cut.shape[1]))
        reconstructed_whole_image[large_image.shape[0]-256:, int(remaining_width/2):int(large_image.shape[1]-remaining_width/2)] = reconstructed_image_borde_abajo

        #aplico prediction al borde lateral
        patches_borde_lateral = patchify(large_image[int(remaining_height/2):int(large_image.shape[0]-remaining_height/2),large_image.shape[1]-256:], (256, 256), step=256)  #Step=256 for 256 patches means no overlap
        predicted_patches_borde_lateral = []
        for i in range(patches_borde_lateral.shape[0]):
            for j in range(patches_borde_lateral.shape[1]):
                
                single_patch = patches_borde_lateral[i,j,:,:]
                if single_patch.shape==(256,256):
                    
                    single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
                    single_patch_shape = single_patch_norm.shape[:2]
                    single_patch_norm=single_patch_norm[:,:,0][:,:,None]
                    single_patch_input=np.expand_dims(single_patch_norm, 0)
                    single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.2).astype(np.uint8)      
                    predicted_patches_borde_lateral.append(cv2.resize(single_patch_prediction, single_patch_shape[::-1]))

        predicted_patches_borde_lateral = np.array(predicted_patches_borde_lateral)

        predicted_patches_borde_lateral_reshaped = np.reshape(predicted_patches_borde_lateral, (patches_borde_lateral.shape[0], patches_borde_lateral.shape[1], 256,256) )

        reconstructed_image_borde_lateral = unpatchify(predicted_patches_borde_lateral_reshaped, (large_image_cut.shape[0],256))
        reconstructed_whole_image[int(remaining_height/2):int(large_image.shape[0]-remaining_height/2),large_image.shape[1]-256:] = reconstructed_image_borde_lateral
        reconstructed_whole_image[reconstructed_whole_image == 1] = 255  # Cambiar los píxeles con valor 1 a 255
        kernel_morf = np.ones((5,5), np.uint8)
        eroded_image = cv2.erode(reconstructed_whole_image, kernel_morf, iterations=1)
        reconstructed_whole_image = cv2.dilate(eroded_image, kernel_morf, iterations=1)
        segm_images.append(reconstructed_whole_image)
        
        
    final_segm_image = np.array(segm_images).astype(np.uint8)   
    path_mitocondria= os.path.join(project_dir, 'mascara_mitocondrias.tif')
    #tifffile.imsave(path_mitocondria, np.transpose(final_segm_image, axes=(0, 2, 1))) #DEJAR ESTE PARA FINAL
    tifffile.imsave(path_mitocondria, final_segm_image)
    return final_segm_image
        
# parent_path=r'\Users\ulabceriani\Documents\ITBA'
# volumen=tifffile.imread(r'\Users\ulabceriani\Documents\ITBA\pruebainterfaz\prueba_zt22\multiplied.tif')
# project_dir=r'\Users\ulabceriani\Documents\ITBA\resultados_multiclass_unet'
# print(volumen.shape)
# segmentacion_mitocondrias=predict_mito_resattunet(volumen, parent_path, project_dir)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        