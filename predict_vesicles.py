#predict_vesicles
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

def predict_vesicles_bin5(volumen, parent_path, project_dir,shape_bin5): #parent_path: carpeta con todas las funciones, project_dir: carpeta con volumen, mascara neuronas, mascada mito, etc.
    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #pesos para bin1

    PATH = os.path.join(parent_path, 'model_dir_bin5\epoch-9.pth')  #r'\Users\ulabceriani\Documents\ITBA\model_dir\epoch-24.pth'

    PATH_post = os.path.join(parent_path, 'model_dir2_bin5\epoch-16.pth') #r'\Users\ulabceriani\Documents\ITBA\model_dir_2\epoch-54.pth'

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

    sliding_size = 10
    window_size = 100


    tiff=volumen
    print(tiff.shape)
    z=tiff.shape[0]
    y=tiff.shape[1]
    x=tiff.shape[2]

    tiff_prob_maps=np.zeros((z,y*10,x*10))
    tiff_centers_semifinal=np.zeros((z,y*10,x*10))
    tiff_prob_colors=np.zeros((z,y*10,x*10))
    tiff_centers_final=np.zeros((z,y*10,x*10))

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
        nuevo_ancho = 10 * np_img.shape[1]
        nuevo_alto = 10 * np_img.shape[0] 

        # Redimensionar la imagen usando interpolación bilineal
        imagen_con_mas_resolucion = cv2.resize(np_img, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LINEAR)
        def GaussianKernel2D(nfil, ncol, sigma):
            return(np.dot(cv2.getGaussianKernel(nfil, sigma), np.transpose(cv2.getGaussianKernel(ncol, sigma))))
        kernel_FG = GaussianKernel2D(3,3,1)
        imagen_Gauss_FiltradaCv2 =cv2.filter2D(imagen_con_mas_resolucion,-1,kernel_FG)
        img=imagen_Gauss_FiltradaCv2
        np_img = np.array(img)
        
        
        imagen_byn=np.zeros_like(np_img)
        imagen_byn[np_img>2]=255
        imagen_byn = cv2.convertScaleAbs(imagen_byn)

        
        contornos, _ = cv2.findContours(imagen_byn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contornos)==0: 
            continue

        # Inicializar variables para los puntos más extremos
        x_min = imagen_byn.shape[1]
        x_max = 0
        y_min = imagen_byn.shape[0]
        y_max = 0

        # Recorrer todos los contornos para encontrar los puntos extremos
        for contorno in contornos:
            for punto in contorno:
                x, y = punto[0]
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y

        #print(x_min,x_max,y_min,y_max)

        imagen_recortada=np_img[y_min-50:y_max+50,x_min-50:x_max+50]
         
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
                    p_map[int((y + 50) / sliding_size),
                                int((x + 50) / sliding_size)] = pvalue

                
                patch_counter += 1

        print(patch_counter)
        proc_pmap = cv2.resize(p_map, (np_img.shape[1], np_img.shape[0])) #que quede del mismo tamaño de la imagen
        proc_pmap = cv2.blur(proc_pmap, (3, 3))

        if np.max(proc_pmap) > 0:
            proc_pmap = (proc_pmap / (np.max(proc_pmap))) * 255

        # set a threshold for proc_map (below 20% of 255, pixel=0)
        for xx in range(proc_pmap.shape[0]):
            for yy in range(proc_pmap.shape[1]):
                if proc_pmap[xx, yy] < 255 / 100 * 20:
                    proc_pmap[xx, yy] = 0
        
        
        np_img=np.array(img)
        prob_map_final=np.zeros_like(np_img)
        prob_map_final[y_min-50:y_max+50,x_min-50:x_max+50]=proc_pmap

        tiff_prob_maps[i,:,:]=prob_map_final


        #etiquetado y centros (cle39)

        map=np.zeros_like(np_img)
        print("map: ",map.shape)
        map[prob_map_final>=7]=1
        # Encontrar objetos conectados
        labelarray, num_features = label(map)
        print(num_features)
        # Contar los píxeles en cada objeto conectado
        counts = measurements.sum(map, labelarray, index=np.arange(labelarray.max() + 1))
        # Umbral de tamaño mínimo
        min_size = 340

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
        #tiff_centers_semifinal[i,:,:]=relevant_maxima_array

        #Step 4: Separate maxima locations into labels using masked voronoi
        voronoi_separation = cle.masked_voronoi_labeling(img_relevant_maxima, img_thresh)
        print(type(voronoi_separation))
        print(voronoi_separation.shape)
        voronoi_separation_np = cle.pull(voronoi_separation)
        voronoi_separation = np.array(voronoi_separation_np)
        #voronoi_separation[voronoi_separation>=30]=255

        voronoi_separation = voronoi_separation.astype(np.float32)
        print(type(voronoi_separation))
        print(voronoi_separation.shape)
        tiff_prob_colors[i,:,:]=voronoi_separation
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

        window_size_post=200

        # put padding on image
        np_img = np.array(img)
        if len(np_img.shape) > 2:
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)

        imagen_res=np.zeros_like(np_img)

        np_img_padded = np.zeros((np_img.shape[0] + 200, np_img.shape[1] + 200))
        np_img_padded[100:np_img.shape[0] + 100,
                        100:np_img.shape[1] + 100] = np_img

        # iterate over the detected vesicles
        for det_ves in range(len(x_labels)):
            snapshot = np_img_padded[int(y_labels[det_ves]):
                                        int(y_labels[det_ves]) + 200,
                                        int(x_labels[det_ves]):
                                            int(x_labels[det_ves]) + 200]
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

            #cv2.imwrite(f"/Users/Milagros/Downloads/posibles_vesiculas/{preds}_snapshot{det_ves}.jpg", recorte)

            if preds == 1:
                x_labels_semifinal.append(x_labels[det_ves])
                y_labels_semifinal.append(y_labels[det_ves])

        print(len(x_labels_semifinal))

        for j in range(len(x_labels_semifinal)):
            imagen_res[int(y_labels_semifinal[j]),int(x_labels_semifinal[j])]=255
            imagen_res[int(y_labels_semifinal[j])+1,int(x_labels_semifinal[j])]=255
            imagen_res[int(y_labels_semifinal[j])+1,int(x_labels_semifinal[j])+1]=255
            imagen_res[int(y_labels_semifinal[j])+1,int(x_labels_semifinal[j])-1]=255
            imagen_res[int(y_labels_semifinal[j])-1,int(x_labels_semifinal[j])]=255
            imagen_res[int(y_labels_semifinal[j])-1,int(x_labels_semifinal[j])-1]=255
            imagen_res[int(y_labels_semifinal[j])-1,int(x_labels_semifinal[j])+1]=255
            imagen_res[int(y_labels_semifinal[j]),int(x_labels_semifinal[j])+1]=255
            imagen_res[int(y_labels_semifinal[j]),int(x_labels_semifinal[j])-1]=255
            slide_resize_center_final[int((y_labels_semifinal[j])/10),int((x_labels_semifinal[j])/10)]=255

        #tiff_centers_final[i,:,:]=imagen_res
        #resized_img = cv2.resize(imagen_res, (954,2802), interpolation=cv2.INTER_NEAREST_EXACT)
        #resize_centers_final.append(resized_img)
        resize_centers_final.append(slide_resize_center_final)


           

    #tiff_centers_final=np.array(tiff_centers_final).astype(np.uint8) #tiff centers final blancos
    #tiff_centers_semifinal=np.array(tiff_centers_semifinal).astype(np.uint8)
    #tiff_prob_colors=np.array(tiff_prob_colors).astype(np.uint8)
    #tiff_prob_maps=np.array(tiff_prob_maps).astype(np.uint8)

    #path_vesicle_center= os.path.join(project_dir, 'mascara_centro_vesiculas.tif')
    #tifffile.imsave(path_vesicle_center, tiff_centers_final)

    #tifffile.imsave(r'\Users\ulabceriani\Documents\ITBA\ITBA_ZT2\prueba_todo_junto\prob_colors.tif',tiff_prob_colors)

    #resizear de bin 1 a bin5 (/5)
    #tiff_centers_final=cv2.resize(tiff_centers_final,(tiff_centers_final.shape[0]/5,tiff_centers_final.shape[1]/5))
    #tiff_prob_colors=cv2.resize(tiff_prob_colors,(tiff_prob_colors.shape[0]/5,tiff_prob_colors.shape[1]/5))
    resize_centers_final = np.array(resize_centers_final).astype(np.uint8)
    resize_prob_maps= np.array(resize_prob_maps).astype(np.uint8)
    #path_area_vesicles_bin1= os.path.join(project_dir, 'mascara_area_vesicles_bin1.tif')
    #tifffile.imsave(path_area_vesicles_bin1, np.transpose(tiff_prob_colors, axes=(0,2,1)))
    return resize_prob_maps,resize_centers_final # colores  #nunca se muestra el prob colors hasta que no sea el final

