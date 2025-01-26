import numpy as np
from tvtk.api import tvtk
from tvtk.common import configure_input
import tifffile as tiff
import cv2
import pandas as pd
import os

# Función para encontrar el volumen y el área superficial usando TVTK
def vtk_volume(data, spacing=(1, 1, 1), origin=(0, 0, 0)):
    data[data == 0] = -1
    grid = tvtk.ImageData(spacing=spacing, origin=origin)
    grid.point_data.scalars = data.T.ravel()  # Transponer la data
    grid.point_data.scalars.name = 'scalars'
    grid.dimensions = data.shape

    iso = tvtk.ImageMarchingCubes()
    configure_input(iso, grid)
    mass = tvtk.MassProperties()
    configure_input(mass, iso)
    
    return mass.volume, mass.surface_area

# Función para calcular métricas por cada mitocondria etiquetada
def metrics_mitocondria(mito_tag_volume, pixel_size, z_ratio):
    max_label = mito_tag_volume.max()
    volumen_mito = np.zeros(max_label + 1)
    surface_mito = np.zeros(max_label + 1)
    mbi = np.zeros(max_label + 1)

    for label in range(1, max_label + 1):
        mito_mask = (mito_tag_volume == label).astype(np.int8)
        mito_mask = np.transpose(mito_mask, (2, 1, 0))
        print(mito_mask.shape)
        # Llamada a vtk_volume para cada mitocondria etiquetada
        if mito_mask.any():
            volume, surface_area = vtk_volume(mito_mask, spacing=(pixel_size, pixel_size, z_ratio*pixel_size))
            volumen_mito[label] = volume
            surface_mito[label] = surface_area
            print(volume, surface_area)
            # Calcular longitud transversal
            xy_positions = np.column_stack(np.where(mito_mask.max(axis=0) == 1))
            if xy_positions.shape[0] > 0:
                max_xy_distance = np.max(np.linalg.norm(xy_positions - xy_positions[:, None], axis=2))
            else:
                max_xy_distance = 0
            
            mbi[label] = max_xy_distance * pixel_size

    # Calcular el índice de ramificación mitocondrial (MBI)
    mci = np.zeros_like(volumen_mito)
    for i in range(1, len(volumen_mito)):
        mci[i] = (surface_mito[i]**3) / (16 * np.pi**2 * volumen_mito[i]**2)

        # Calcular la longitud longitudinal (extensión en el eje Z)
        mito_positions_z = np.where(mito_tag_volume == i)[0]
        if mito_positions_z.size > 0:
            length_z = (np.max(mito_positions_z) - np.min(mito_positions_z) + 1) * z_ratio * pixel_size
            mbi[i] /= length_z if length_z != 0 else np.nan  # Dividir la longitud transversal por la longitud longitudinal

    return volumen_mito, surface_mito, mci, mbi

# Cargar las máscaras TIFF
def load_tiff_mask(file_path):
    mito_tag_volume = tiff.imread(file_path)
    return mito_tag_volume


tiff_file = r'\Users\ulabceriani\Documents\ITBA\pruebainterfaz\prueba_zt22\mascara_mitocondrias_etiquetada_hasta450.tif'
path_carpeta= r'\Users\ulabceriani\Documents\ITBA\pruebainterfaz\prueba_zt22'

# Tamaño de píxel y proporción Z
pixel_size = 25  # Ajusta según tus datos
z_ratio = 2.4  # Ajusta según tus datos

# Cargar las máscaras etiquetadas
mito_tag_volume = load_tiff_mask(tiff_file)

def calculate_metrics(mascara_etiquetada_mitochondria, pixel_size, z_scale):
    volume_mito, surface_mito, mci, mbi = metrics_mitocondria(mascara_etiquetada_mitochondria, pixel_size, z_scale)
    output_path = os.path.join(path_carpeta, 'new_metrics_zt22.xlsx')
        # Prepare the data for the DataFrame
    data = {
        'Mitochondria': [f'Mitochondria {i}' for i in range(1,len(volume_mito))],
        'Volume [nm3]': volume_mito[1:],
        'Surface Area [nm2]': surface_mito[1:],
        'MCI': mci[1:],
        'MBI': mbi[1:]
    }
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)
    print('se ha guardado el excel con las metricas')

calculate_metrics(mito_tag_volume[:,:,:], pixel_size, z_ratio)