import tifffile
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

# Cargar el volumen de la máscara
path_mask = r'\Users\ulabceriani\Documents\ITBA\ITBA_ZT22\MASK_MITO_zt22.tif'
mask_mito = tifffile.imread(path_mask)

# Crear un nuevo volumen para la máscara procesada
processed_mask = np.zeros_like(mask_mito)

# Recorrer cada slice y procesar
for i in range(mask_mito.shape[0]):
    slice_mito = mask_mito[i]
    # Mantener solo las partes blancas
    processed_mask[i] = np.where(slice_mito == 255, 255, 0)

# Guardar el nuevo volumen procesado
processed_mask=np.array(processed_mask).astype(np.uint8)
path_processed_mask = r'\Users\ulabceriani\Documents\ITBA\ITBA_ZT22\MASK_mito_zt22.tif'
import tifffile
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

# Cargar el volumen de la máscara
path_mask = r'\Users\ulabceriani\Documents\ITBA\ITBA_ZT22\MASK_MITO_zt22.tif'
mask_mito = tifffile.imread(path_mask)

processed_mask = np.zeros_like(mask_mito)
print(mask_mito.shape)

# Recorrer cada slice y procesar
for i in range(mask_mito.shape[0]):
    slice_mito = mask_mito[i]
    # Identificar los píxeles donde todos los canales son 255 usando np.where
    white_pixels = np.where((slice_mito[:, :, 0] == 255) & 
                            (slice_mito[:, :, 1] == 255) & 
                            (slice_mito[:, :, 2] == 255))
    
    # Asignar el valor 255 a los píxeles blancos en la máscara procesada
    processed_mask[i][white_pixels] = [255, 255, 255]
    
    # Visualización de ejemplo para la cuarta slice
    if i == 3:
        plt.figure(figsize=(15, 15))
        plt.imshow(processed_mask[i])
        plt.show()

for i in range(processed_mask.shape[0]):
    processed_mask[i] = cv2.cvtColor(processed_mask[i], cv2.COLOR_RGB2GRAY)

# Guardar el nuevo volumen procesado
processed_mask=np.array(processed_mask).astype(np.uint8)
path_processed_mask = r'\Users\ulabceriani\Documents\ITBA\ITBA_ZT22\MASK_mito_zt22.tif'
try:
    tifffile.imwrite(path_processed_mask, processed_mask)
    print('Processed mask saved successfully.')
except Exception as e:
    print(f'Error saving processed mask: {e}')
