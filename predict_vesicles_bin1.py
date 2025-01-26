#predict_vesicles_bin1
import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from CNNs_GaussianNoiseAdder import MultiClass, MultiClassPost
import tifffile
from scipy.ndimage import measurements, label
import cv2
import os
from skimage.io import imsave
import pyclesperanto_prototype as cle
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import matplotlib as plt

def predict_vesicles(volumen, parent_path, project_dir, shape_bin5, shape_bin1): #parent_path: carpeta con todas las funciones, project_dir: carpeta con volumen, mascara neuronas, mascada mito, etc.
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #pesos para bin1

    PATH = os.path.join(parent_path, 'model_dir\epoch-24.pth')  #r'\Users\ulabceriani\Documents\ITBA\model_dir\epoch-24.pth'

    PATH_post = os.path.join(parent_path, 'model_dir_2\epoch-54.pth') #r'\Users\ulabceriani\Documents\ITBA\model_dir_2\epoch-54.pth'

    if torch.cuda.is_available():
        model = MultiClass(out=2).to(device)
        model.load_state_dict(torch.load(PATH))
        model_post = MultiClassPost(out=2).to(device)
        model_post.load_state_dict(torch.load(PATH_post))
    else:
        model = MultiClass(out=2)
        model.load_state_dict(torch.load(PATH, map_location=device))
        model_post = MultiClassPost(out=2)
        model_post.load_state_dict(torch.load(PATH_post,map_location=device))

    model.eval()
    model_post.eval()

    sliding_size = 5
    window_size = 50


    tiff=volumen
    print(tiff.shape)

    #tiff_prob_maps=np.zeros_like(tiff, dtype=np.uint8)
    tiff_centers_semifinal=np.zeros_like(tiff, dtype=np.uint8)
    tiff_prob_colors=np.zeros_like(tiff, dtype=np.uint16)
    tiff_centers_final=np.zeros_like(tiff, dtype=np.uint8)

    resize_prob_maps=[]
    resize_centers_final=[]


    for i in range((tiff.shape[0])):
        print("i:",i)
        slide_resize_center_final=np.zeros(shape_bin5)

        img=tiff[i,:,:]
        np_img = np.array(img)
        if len(np_img.shape) > 2:
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        print(np_img.shape)
        
        imagen_byn=np.zeros_like(np_img)
        imagen_byn[np_img>2]=255
        imagen_byn = cv2.convertScaleAbs(imagen_byn)

        prob_map_final=np.zeros_like(np_img)

        contornos, _ = cv2.findContours(imagen_byn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contornos)==0: 
            resize_centers_final.append(slide_resize_center_final)
            resized_prob=np.zeros(shape_bin5)
            resize_prob_maps.append(resized_prob)
            continue
        
        else:
            ancho_minimo = 5
            alto_minimo = 5

            # Filtrar contornos según el tamaño
            contornos_filtrados = []
            for contorno in contornos:
                x, y, ancho, alto = cv2.boundingRect(contorno)
                
                # Solo agregar los contornos que superen el tamaño mínimo
                if ancho >= ancho_minimo and alto >= alto_minimo:
                    contornos_filtrados.append(contorno)
            contornos=contornos_filtrados
            print(len(contornos))
            # Recorrer todos los contornos
            for contorno in contornos:
                # Obtener las coordenadas del rectángulo delimitador para cada contorno
                x_min, y_min, ancho, alto = cv2.boundingRect(contorno)
                #print(x_min,y_min,ancho, alto)

                x_max = x_min + ancho
                y_max = y_min + alto

                # Asegurarse de que no se excedan los límites de la imagen al aplicar el margen
                x_min = max(x_min - 5, 0)
                y_min = max(y_min - 5, 0)
                x_max = min(x_max + 5, imagen_byn.shape[1])
                y_max = min(y_max + 5, imagen_byn.shape[0])

                # Recortar la imagen utilizando las coordenadas del rectángulo
                imagen_recortada = np_img[y_min:y_max, x_min:x_max]
                #plt.imshow(imagen_recortada, cmap ='gray')
                #plt.show()
                np_img=imagen_recortada

                p_map = np.zeros((int(np_img.shape[0] / sliding_size),
                            int(np_img.shape[1] / sliding_size)))
        
                patch_counter = 0

                for x in range(0, np_img.shape[1], sliding_size):

                    # iterate over image.shape[0] in steps of size == sliding_size
                    for y in range(0, np_img.shape[0], sliding_size):
                        snapshot = np_img[y :y + window_size,
                                                    x :x + window_size]
                        patch=snapshot

                        if (snapshot.shape[0] != window_size) or (
                                snapshot.shape[1] != window_size):
                            continue
                        snapshot=cv2.resize(snapshot, (40, 40))
                        snapshot = snapshot.reshape(1, 40, 40)
                        if np.max(snapshot) != np.min(snapshot):
                            snapshot = (snapshot - np.min(snapshot)) / (
                                np.max(snapshot) - np.min(snapshot))
                        snapshot = (snapshot - 0.5) / 0.5
                        snapshot = torch.from_numpy(snapshot)
                        snapshot = snapshot.unsqueeze(0)

                        if torch.cuda.is_available():
                            output = model.forward(snapshot.float().cuda())
                            valuemax, preds = torch.max(output, 1)
                            valuemin, _ = torch.min(output, 1)
                            valuemax = valuemax.cpu()
                            valuemin = valuemin.cpu()
                            preds = preds.cpu()
                        else:
                            output = model.forward(snapshot.float())
                            valuemax, preds = torch.max(output, 1)
                            valuemin, _ = torch.min(output, 1)
                        
                        if preds == 1:
                            valuemax = valuemax.data.numpy()
                            valuemin = valuemin.data.numpy()
                            pvalue = np.exp(valuemax) / (np.exp(valuemax) + np.exp(
                                valuemin))
                            p_map[int((y + 25) / sliding_size),
                                        int((x + 25) / sliding_size)] = pvalue

                        
                        patch_counter += 1


                print(patch_counter)
                print((np_img.shape[1], np_img.shape[0]))
                proc_pmap = cv2.resize(p_map, (np_img.shape[1], np_img.shape[0])) #que quede del mismo tamaño de la imagen
                proc_pmap = cv2.blur(proc_pmap, (3, 3))

                if np.max(proc_pmap) > 0:
                    proc_pmap = (proc_pmap / (np.max(proc_pmap))) * 255

                # set a threshold for proc_map (below 20% of 255, pixel=0)
                for xx in range(proc_pmap.shape[0]):
                    for yy in range(proc_pmap.shape[1]):
                        if proc_pmap[xx, yy] < 255 / 100 * 20:
                            proc_pmap[xx, yy] = 0
                
                
                
                prob_map_final[y_min:y_max, x_min:x_max] = np.clip(prob_map_final[y_min:y_max, x_min:x_max] + proc_pmap, 0, 255)
                np_img = np.array(img)
                if len(np_img.shape) > 2:
                    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        
        

        np_img=np.array(img)

        #etiquetado y centros (cle39)

        map=np.zeros_like(np_img)
        map[prob_map_final>=7]=1
        # Encontrar objetos conectados
        labelarray, num_features = label(map)
        print(num_features)
        # Contar los píxeles en cada objeto conectado
        counts = measurements.sum(map, labelarray, index=np.arange(labelarray.max() + 1))
        # Umbral de tamaño mínimo
        min_size = 170

        filtered_map = map.copy()

        # Iterar sobre cada objeto conectado y eliminar los que son más pequeños que el umbral
        for k, count in enumerate(counts):
            if count < min_size:
                filtered_map[labelarray == k] = 0  # Eliminar objetos pequeños

        img_filtered_map=filtered_map

        # List all available devices
        all_devices = cle.available_device_names()
        print("Available devices:", all_devices)

        # Select the best device (example: an RTX GPU)
        selected_device = cle.select_device('RTX')
        print("Selected device:", selected_device)
        # select a specific OpenCL / GPU device and see which one was chosen
        cle.select_device('RTX')

        img_gpu=cle.push(img_filtered_map)
        print("Image size in GPU: " + str(img_filtered_map.shape))
        
        


        # Step 1: heavy gaussian blur the image (e.g., sigma=12) and detect maxima for each nuclei
        # heavy gaussian blurring assists in detecting maxima that reflects the objects.
        #If objects are closer, you may want to decrease the amount of blurring.
        img_gaussian = cle.gaussian_blur(img_gpu, sigma_x=5, sigma_y=5, sigma_z=5)
        #cle.imshow(img_gaussian, color_map='gray')

        # Find out the maxima locations for each 'blob'
        img_maxima_locations = cle.detect_maxima_box(img_gaussian, radius_x=0, radius_y=0, radius_z=0)

        #Number of maxima locations (= number of objects)
        #This number depend on the amount of Gaussian blur
        number_of_maxima_locations = cle.sum_of_all_pixels(img_maxima_locations)
        print("number of detected maxima locations", number_of_maxima_locations)


        #Step 2: threshold the input image after applying light gaussian blur (sigma=1)
        img_gaussian2 = cle.gaussian_blur(img_filtered_map, sigma_x=1, sigma_y=1, sigma_z=1)
        img_thresh = cle.threshold_otsu(img_gaussian2)

        #Step 3: Exclude maxima locations from the background, to make sure we only include the ones from nuclei
        # We can do this by using binary and operation
        img_relevant_maxima = cle.binary_and(img_thresh, img_maxima_locations)

        number_of_relevant_maxima_locations = cle.sum_of_all_pixels(img_relevant_maxima)
        print("number of relevant maxima locations", number_of_relevant_maxima_locations)
        relevant_maxima_array = cle.pull(img_relevant_maxima)
        relevant_maxima_array=255*relevant_maxima_array
        #Wrtie image as tif. Ue imageJ for visualization
        relevant_maxima_array=np.array(relevant_maxima_array)
        tiff_centers_semifinal[i,:,:]=relevant_maxima_array
        

        #Step 4: Separate maxima locations into labels using masked voronoi
        voronoi_separation = cle.masked_voronoi_labeling(img_relevant_maxima, img_thresh)
        tiff_prob_colors[i,:,:]=voronoi_separation
        print(type(voronoi_separation))
        print(voronoi_separation.shape)
        
        
        voronoi_separation_np = cle.pull(voronoi_separation)
        voronoi_separation = np.array(voronoi_separation_np)
        #voronoi_separation[voronoi_separation>=30]=255

        voronoi_separation = voronoi_separation.astype(np.float32)
        resized_prob = cv2.resize(voronoi_separation, ((shape_bin5[1],shape_bin5[0])), interpolation=cv2.INTER_NEAREST_EXACT)
        resize_prob_maps.append(resized_prob)

        #segundo classifier
        white_pixels = np.where((relevant_maxima_array >=253))

        # Extraer las coordenadas X y Y
        x_labels = white_pixels[1]
        y_labels = white_pixels[0]

        print("xlabels=",len(x_labels))

        #SEGUNDO MODELO
        x_labels_semifinal = []
        y_labels_semifinal = []

        window_size_post=100

        # put padding on image
        np_img = np.array(img)
        if len(np_img.shape) > 2:
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)

        imagen_res=np.zeros_like(np_img)

        np_img_padded = np.zeros((np_img.shape[0] + 100, np_img.shape[1] + 100))
        np_img_padded[50:np_img.shape[0] + 50,
                        50:np_img.shape[1] + 50] = np_img

        # iterate over the detected vesicles
        for det_ves in range(len(x_labels)):
            snapshot = np_img_padded[int(y_labels[det_ves]):
                                        int(y_labels[det_ves]) + 100,
                                        int(x_labels[det_ves]):
                                            int(x_labels[det_ves]) + 100]
            if (snapshot.shape[0] != window_size_post) or (
                    snapshot.shape[1] != window_size_post):
                continue

            recorte=snapshot

            snapshot=cv2.resize(snapshot, (80, 80))
            snapshot = snapshot.reshape(1, snapshot.shape[0],
                                        snapshot.shape[1])
            if np.max(snapshot) != np.min(snapshot):
                snapshot = (snapshot - np.min(snapshot)) / (
                    np.max(snapshot) - np.min(snapshot))
            snapshot = (snapshot - 0.5) / 0.5
            snapshot = torch.from_numpy(snapshot)
            snapshot = snapshot.unsqueeze(0)

            # feed image patches into the second (refinement) classifier
            if torch.cuda.is_available():
                output = model_post.forward(snapshot.float().cuda())
                valuemax, preds = torch.max(output, 1)
                preds = preds.cpu()

            else:
                output = model_post.forward(snapshot.float())
                valuemax, preds = torch.max(output, 1)

            #cv2.imwrite(r'C:\Users\ulabceriani\Documents\ITBA\pruebainterfaz\prueba_zt22_100slices\second_classifier_predictions\segunda_prueba\slide{}_{}_snapshot{}.jpg'.format(i,preds, det_ves), recorte)

            if preds == 1:
                x_labels_semifinal.append(x_labels[det_ves])
                y_labels_semifinal.append(y_labels[det_ves])

        print(len(x_labels_semifinal))

        for j in range(len(x_labels_semifinal)):
            imagen_res[int(y_labels_semifinal[j]),int(x_labels_semifinal[j])]=255
            # imagen_res[int(y_labels_semifinal[j])+1,int(x_labels_semifinal[j])]=255
            # imagen_res[int(y_labels_semifinal[j])+1,int(x_labels_semifinal[j])+1]=255
            # imagen_res[int(y_labels_semifinal[j])+1,int(x_labels_semifinal[j])-1]=255
            # imagen_res[int(y_labels_semifinal[j])-1,int(x_labels_semifinal[j])]=255
            # imagen_res[int(y_labels_semifinal[j])-1,int(x_labels_semifinal[j])-1]=255
            # imagen_res[int(y_labels_semifinal[j])-1,int(x_labels_semifinal[j])+1]=255
            # imagen_res[int(y_labels_semifinal[j]),int(x_labels_semifinal[j])+1]=255
            # imagen_res[int(y_labels_semifinal[j]),int(x_labels_semifinal[j])-1]=255
            slide_resize_center_final[int((y_labels_semifinal[j])/5),int((x_labels_semifinal[j])/5)]=255

        tiff_centers_final[i,:,:]=imagen_res
        #resized_img = cv2.resize(imagen_res, (954,2802), interpolation=cv2.INTER_LINEAR)
        resize_centers_final.append(slide_resize_center_final)

           
    #tiff_centers_final=np.array(tiff_centers_final).astype(np.uint8) #tiff centers final blancos
    #tiff_centers_semifinal=np.array(tiff_centers_semifinal).astype(np.uint8)
    #tiff_prob_colors=np.array(tiff_prob_colors).astype(np.uint8)
    #tiff_prob_maps=np.array(tiff_prob_maps).astype(np.uint8)

    #path_vesicle_center= os.path.join(project_dir, 'mascara_centro_vesiculas.tif')
    #tifffile.imsave(path_vesicle_center, tiff_centers_final)

    #tifffile.imsave(r'\Users\ulabceriani\Documents\ITBA\ITBA_ZT2\prueba_todo_junto\prob_colors.tif',tiff_prob_colors)
    #tifffile.imsave(r'\Users\ulabceriani\Documents\ITBA\pruebainterfaz\prueba_zt22\predictions_bin1\center_semifinal.tif', tiff_centers_semifinal)
    #tifffile.imsave(r'\Users\ulabceriani\Documents\ITBA\pruebainterfaz\prueba_zt22\predictions_bin1\pmap_inicial_blanco.tif', prob_map_final)
    tifffile.imsave(r'\Users\ulabceriani\Documents\ITBA\pruebainterfaz\prueba_zt22\predictions_bin1\prob_colors.tif', np.transpose(tiff_prob_colors,  axes=(0,2,1)))
    tifffile.imsave(r'\Users\ulabceriani\Documents\ITBA\pruebainterfaz\prueba_zt22\predictions_bin1\center_final.tif', np.transpose(tiff_centers_final, axes=(0,2,1)))
    #resizear de bin 1 a bin5 (/5)
    #tiff_centers_final=cv2.resize(tiff_centers_final,(tiff_centers_final.shape[0]/5,tiff_centers_final.shape[1]/5))
    #tiff_prob_colors=cv2.resize(tiff_prob_colors,(tiff_prob_colors.shape[0]/5,tiff_prob_colors.shape[1]/5))
    resize_centers_final = np.array(resize_centers_final).astype(np.uint8)
    resize_prob_maps= np.array(resize_prob_maps).astype(np.uint8)
    #path_area_vesicles_bin1= os.path.join(project_dir, 'mascara_area_vesicles_bin1.tif')
    #tifffile.imsave(path_area_vesicles_bin1, np.transpose(tiff_prob_colors, axes=(0,2,1)))
    return resize_prob_maps,resize_centers_final #cambiar despues # colores  #nunca se muestra el prob colors hasta que no sea el final

def region_growing(image, seeds, int_range, p, grafico):
    height, width = image.shape
    region = np.zeros_like(image)

    def is_in_range(pixel):
        return int_range[0] <= image[pixel[0], pixel[1]] <= int_range[1]

    if p == 8:
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    elif p == 4:
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    else:
        raise ValueError("p debe ser 4 u 8")

    for seed in seeds:
        seed_points = [seed]
        while seed_points:
            current_seed = seed_points.pop()
            region[current_seed[0], current_seed[1]] = 255

            for neighbor in neighbors:
                x, y = current_seed[0] + neighbor[0], current_seed[1] + neighbor[1]
                if 0 <= x < height and 0 <= y < width and region[x, y] == 0 and is_in_range((x, y)):
                    seed_points.append((x, y))

    if grafico:
        plt.imshow(region, cmap="gray", vmin=0, vmax=255)
        plt.title("Imagen Region Growing")
        plt.show()

    return region

def extract_regions(centers, pmap, p, grafico):
    result_img = np.zeros_like(pmap)
    result_img_blanco = np.zeros_like(pmap)
    points = np.argwhere(centers >= 1)

    for point in points:
        x, y = point[1], point[0]
        seeds = [(y, x)]  # Ajuste para el formato de las coordenadas
        intensity=pmap[y,x]
        int_range=[intensity - 0.5, intensity + 0.5]
        mask = region_growing(pmap, seeds, int_range, p, grafico=False)
        result_img_blanco[mask == 255] = 255
        result_img[mask == 255] = centers[y,x]

    if grafico:
        plt.imshow(result_img, cmap="gray", vmin=0, vmax=255)
        plt.title("Imagen Resultante")
        plt.show()

    return result_img, result_img_blanco

def etiquetado_vesiculas_unicas(center_final, prob_colors, project_dir):
    ##centers etiquetados
    volumen_center_etiquetado=np.zeros_like(center_final)
    threshold=5
    max_label = 0
    for i in range(center_final.shape[0]):
        slide = center_final[i, :, :]
        print(np.max(slide))
        coordinates = np.column_stack(np.where(slide == 255))
        print(len(coordinates))

        if i == 0:
            # Etiqueta inicial para la primera diapositiva
            for idx, (y, x) in enumerate(coordinates):
                volumen_center_etiquetado[i, y, x] = idx + 1
            max_label = len(coordinates)
        else:
            etiquetado_anterior = volumen_center_etiquetado[i - 1, :, :]
            coordinates_anterior = np.column_stack(np.where(etiquetado_anterior > 0))

            if coordinates_anterior.size > 0:
                # Construir un KDTree para búsqueda rápida de vecinos cercanos
                tree = cKDTree(coordinates_anterior)
                distances, indices = tree.query(coordinates, distance_upper_bound=threshold)

                for idx, (y, x) in enumerate(coordinates):
                    if distances[idx] < threshold:
                        print("repetido")
                        # Si está dentro del umbral, usar la etiqueta del vecino más cercano
                        nearest_y, nearest_x = coordinates_anterior[indices[idx]]
                        volumen_center_etiquetado[i, y, x] = etiquetado_anterior[nearest_y, nearest_x]
                    else:
                        # Si no, asignar una nueva etiqueta
                        max_label += 1
                        volumen_center_etiquetado[i, y, x] = max_label
    print('cantidad de vesiculas: ',max_label)
    print(np.max(volumen_center_etiquetado))
    
    
    #probability map etiquetado final

    tiff_prob_final_color=np.zeros_like(prob_colors)
    tiff_prob_final_blanco=np.zeros_like(prob_colors)

    # Definir el rango de intensidad y conectividad
    for i in range(volumen_center_etiquetado.shape[0]):
        centers=volumen_center_etiquetado[i,:,:]
        pmap=prob_colors[i,:,:]

        p = 8  # Puede ser 4 o 8
        grafico = False  # Mostrar gráficos

        # Extraer las regiones y generar la imagen resultante
        result_img, result_img_blanco = extract_regions(centers, pmap, p, grafico)

        tiff_prob_final_color[i,:,:]=result_img #el etiquetado
        tiff_prob_final_blanco[i,:,:]=result_img_blanco
    
    
    #path_vesicle_area= os.path.join(project_dir, 'mascara_area_vesiculas.tif')
    #tifffile.imsave(path_vesicle_area, tiff_prob_final_blanco)

    return volumen_center_etiquetado, tiff_prob_final_color, tiff_prob_final_blanco, max_label