import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
import tifffile
import io
import pyclesperanto_prototype as cle
from skimage.io import imsave
import os

def mitochondria_labeling_original (volumen):
    # List all available devices
    all_devices = cle.available_device_names()
    print("Available devices:", all_devices)

    # Select the best device (example: an RTX GPU)
    selected_device = cle.select_device('RTX')
    print("Selected device:", selected_device)
    # select a specific OpenCL / GPU device and see which one was chosen
    cle.select_device('RTX')

    input_gpu = cle.push(volumen[:,:,:])
    #print("Image size in GPU: " + str(input_gpu.shape))
    segmented = cle.voronoi_otsu_labeling(input_gpu, spot_sigma=10, outline_sigma=1)
    outline_sigma=1
    print("outlinesigma",outline_sigma)
    segmented_array = cle.pull(segmented)
    #path_to_save=os.path.join(path, 'label')
    #imsave(r'\Users\ulabceriani\Documents\ITBA\ITBA_ZT2\segmented_3d_mito_fullvolume.tif', segmented_array)
    print('finished labeling mitochondria')
    return segmented_array

def mitochondria_labeling(volumen):
    # Asegurarse de que esté usando el dispositivo GPU correcto
    all_devices = cle.available_device_names()
    print("Available devices:", all_devices)
    selected_device = cle.select_device('RTX')
    print("Selected device:", selected_device)
    
    # Convertir el volumen a GPU
    input_gpu = cle.push(volumen[:,:,:])
    
    # Primer etiquetado con sigma 15
    segmented1 = cle.voronoi_otsu_labeling(input_gpu, spot_sigma=15, outline_sigma=1)
    segmented1_array = cle.pull(segmented1)
    print('Finished first labeling with sigma 15')
    
    # Segundo etiquetado con sigma 5
    segmented2 = cle.voronoi_otsu_labeling(input_gpu, spot_sigma=5, outline_sigma=1)
    segmented2_array = cle.pull(segmented2)
    print('Finished second labeling with sigma 5')
    
    # Clonar segmented2 como base
    final_segmented = segmented2_array.copy()
    
    # Encontrar el máximo de etiquetas en segmented1
    max_label2 = np.max(segmented2_array)
    
    # Reemplazar etiquetas donde segmented1 tenga valores
    mask = segmented1_array > 0
    final_segmented[mask] = segmented1_array[mask] + max_label2
    
    print('Finished combining labels')
    return final_segmented