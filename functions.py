import numpy as np
import cv2

def region_growing(image, mask, mask_neurona, seeds, delta, p=8): # p=4 u 8, #range: [100,170]
    # Obtener las dimensiones de la imagen
    height, width = image.shape

    # Direcciones de vecindad 8 (puedes cambiar a 4 si prefieres)
    if p == 8:
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    elif p == 4:
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    else:
        raise ValueError("p debe ser 4 u 8")

    # Para cada punto semilla que se ingresa
    for k in range(len(seeds)):
        coord_semilla = seeds[k]
        seed_points = [coord_semilla]
        print(coord_semilla)
        # Obtener el nivel de gris de la semilla
        seed_value = image[coord_semilla[0], coord_semilla[1]]
        
        def is_in_range(pixel):
            return seed_value - delta <= image[pixel[0], pixel[1]] <= seed_value + delta
        
        while seed_points:
            current_seed = seed_points.pop()
            # Agregar el punto semilla a la región
            mask[current_seed[0], current_seed[1]] = 255

            # Expandir la región verificando los vecinos
            for neighbor in neighbors:
                x, y = current_seed[0] + neighbor[0], current_seed[1] + neighbor[1]
                if (0 <= x < height and 0 <= y < width and mask[x, y] == 0 and 
                    is_in_range((x, y)) and mask_neurona[x, y] != 0):
                    # Agregar el punto vecino a la lista de puntos semilla
                    seed_points.append((x, y))
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask


import numpy as np
import cv2

def region_growing_delete(image, mask, mask_neurona, seeds, p=8):
    # Obtener las dimensiones de la imagen
    height, width = image.shape

    # Direcciones de vecindad 8 (puedes cambiar a 4 si prefieres)
    if p == 8:
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    elif p == 4:
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    else:
        raise ValueError("p debe ser 4 u 8")
    seed=seeds[0]
    if mask[seed[0],seed[1]]==255:
      # Para cada punto semilla que se ingresa
      for k in range(len(seeds)):
          coord_semilla = seeds[k]
          seed_points = [coord_semilla]
          print(coord_semilla)
          
          while seed_points:
              current_seed = seed_points.pop()
              # Agregar el punto semilla a la región
              mask[current_seed[0], current_seed[1]] = 0

              # Expandir la región verificando los vecinos
              for neighbor in neighbors:
                  x, y = current_seed[0] + neighbor[0], current_seed[1] + neighbor[1]
                  if (0 <= x < height and 0 <= y < width and mask[x, y] == 255 and mask_neurona[x, y] != 0):
                      # Agregar el punto vecino a la lista de puntos semilla
                      seed_points.append((x, y))
                      # borrar la posición en la máscara como parte de la región
                      mask[x, y] = 0

    return mask


def region_growing_etiquetado(image, mask, mask_neurona, seeds, delta, cant_ves, p=8): # p=4 u 8, #range: [100,170]
    # Obtener las dimensiones de la imagen
    height, width = image.shape

    # Direcciones de vecindad 8 (puedes cambiar a 4 si prefieres)
    if p == 8:
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    elif p == 4:
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    else:
        raise ValueError("p debe ser 4 u 8")

    # Para cada punto semilla que se ingresa
    for k in range(len(seeds)):
        coord_semilla = seeds[k]
        seed_points = [coord_semilla]
        print(coord_semilla)
        # Obtener el nivel de gris de la semilla
        seed_value = image[coord_semilla[0], coord_semilla[1]]
        
        def is_in_range(pixel):
            return seed_value - delta <= image[pixel[0], pixel[1]] <= seed_value + delta
        
        while seed_points:
            current_seed = seed_points.pop()
            # Agregar el punto semilla a la región
            mask[current_seed[0], current_seed[1]] = cant_ves+1

            # Expandir la región verificando los vecinos
            for neighbor in neighbors:
                x, y = current_seed[0] + neighbor[0], current_seed[1] + neighbor[1]
                if (0 <= x < height and 0 <= y < width and mask[x, y] == 0 and 
                    is_in_range((x, y)) and mask_neurona[x, y] != 0):
                    # Agregar el punto vecino a la lista de puntos semilla
                    seed_points.append((x, y))
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    cant_ves=cant_ves+1
    return mask, cant_ves

def region_growing_etiquetado_delete(image, mask, mask_centers, mask_neurona, seeds, intensidad, p=8): # p=4 u 8, #range: [100,170]
    # Obtener las dimensiones de la imagen
    height, width = image.shape

    # Direcciones de vecindad 8 (puedes cambiar a 4 si prefieres)
    if p == 8:
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    elif p == 4:
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    else:
        raise ValueError("p debe ser 4 u 8")
    

    seed=seeds[0]
    if mask[seed[0],seed[1]]==intensidad:
      # Para cada punto semilla que se ingresa
      for k in range(len(seeds)):
          coord_semilla = seeds[k]
          seed_points = [coord_semilla]
          print(coord_semilla)
          
          while seed_points:
              current_seed = seed_points.pop()
              # Agregar el punto semilla a la región
              mask[current_seed[0], current_seed[1]] = 0
              mask_centers[current_seed[0], current_seed[1]] = 0

              # Expandir la región verificando los vecinos
              for neighbor in neighbors:
                  x, y = current_seed[0] + neighbor[0], current_seed[1] + neighbor[1]
                  if (0 <= x < height and 0 <= y < width and mask[x, y] == intensidad and mask_neurona[x, y] != 0):
                      # Agregar el punto vecino a la lista de puntos semilla
                      seed_points.append((x, y))
                      # borrar la posición en la máscara como parte de la región
                      mask[x, y] = 0
                      mask_centers[x, y] = 0

    return mask, mask_centers




    