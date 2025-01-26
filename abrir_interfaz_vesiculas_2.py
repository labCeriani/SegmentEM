import sys
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QFileDialog, QInputDialog, QMessageBox, QDialog, QPushButton
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QVBoxLayout, QRadioButton, QPushButton, QMessageBox
import pyqtgraph as pg
import tifffile
import cv2
import numpy as np
import shutil
import os
import pandas as pd
from predict_mito import multiply_mask, predict_mito
from predict_neurons import predict_neurons
from predict_vesicles_bin1 import predict_vesicles,etiquetado_vesiculas_unicas
from predict_vesicles import predict_vesicles_bin5
from functions import region_growing, region_growing_delete, region_growing_etiquetado, region_growing_etiquetado_delete
from label_mitochondria import mitochondria_labeling
from Metrics import metrics_mitocondria

class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        loadUi(r'\Users\ulabceriani\Documents\ITBA\pruebainterfaz\prueba_interfaz_vesicles.ui', self)
        self.image_widget.ui.histogram.hide()
        self.image_widget.ui.roiBtn.hide()
        self.image_widget.ui.menuBtn.hide()
        self.current_index = 0
        self.actionNew.triggered.connect(self.create_new_project)
        self.actionOpen.triggered.connect(self.open_project)
        self.actionSave.triggered.connect(self.save_project)
        self.actionMitochondria.triggered.connect(self.predict_mitochondria)
        self.actionNeurons.triggered.connect(self.predict_neurons)
        self.actionVesicles.triggered.connect(self.elegir_metodo_prediccion_ves)
        self.checkBox_neurons.stateChanged.connect(self.opciones_visulalizacion)
        self.checkBox_mito.stateChanged.connect(self.opciones_visulalizacion)

        self.checkBox_center_ves.stateChanged.connect(self.opciones_visulalizacion)
        self.checkBox_area_ves.stateChanged.connect(self.opciones_visulalizacion)

        self.actionMitochondria_Label.triggered.connect(self.label_mitochondria)
        self.actionVesicles_Label.triggered.connect(self.label_ves) #tengo que tener el center vesicle, y el area vesicle con distintas etiquetas
        self.mouse_click_connection = None
        self.actionMitochondria_Add.triggered.connect(self.add_mitochondria)
        self.actionFinish_Add.triggered.connect(self.finish_click)
        self.actionMitochondria_Delete.triggered.connect(self.delete_mitochondria)
        self.actionFinish_Delete.triggered.connect(self.finish_delete)
        self.actionMitochondria_Undo.triggered.connect(self.undo_mitochondria)

        self.actionVesicles_Add.triggered.connect(self.add_vesicles)
        self.actionVesicles_Delete.triggered.connect(self.delete_vesicle)
        self.actionVesicle_Undo.triggered.connect(self.undo_vesicles)

        self.metrics_button.clicked.connect(self.calculate_metrics)
        self.label_tool.setText("")
        self.label_mitocondria.setText("")
        self.label_vesicula.setText("")

    def cargar_imagen(self, path_volumen):
        print("cargar imagen")
        self.tiff=tifffile.imread(path_volumen)
        print("dim tiff: ",self.tiff.shape)
        if self.tiff.ndim != 3:
            # Si no tiene tres dimensiones, cargar todas las páginas y apilarlas
            with tifffile.TiffFile(path_volumen) as tif:
                pages = [page.asarray() for page in tif.pages]
                self.tiff = np.stack(pages, axis=0)  # Apilar las imágenes para crear un array 3D

        self.tiff=np.transpose(self.tiff, axes=(0,2,1))
        print("dim tiff: ",self.tiff.shape)
        #SOLO POR AHORA
        
        #self.tiff=self.tiff[9:,:,:] #PARA ZT14
        #self.tiff=self.tiff[29:,:,:] #PARA ZT2
        #self.tiff=self.tiff[450:,:,:]
        
        self.volumen_a_visualizar=self.tiff
        self.image_widget.getView().setAspectLocked(True)
        self.image_widget.getView().autoRange(padding=0.0)
        self.image_widget.timeLine.setPen((255,255,0,200))
        #self.image_widget.setImage(self.tiff)
        self.volumen_con_mascaras = self.tiff.copy()
        size=os.path.getsize(path_volumen)
        self.GB=np.round(size/10**(9),1)
        self.type=self.tiff.dtype
        self.frames=self.tiff.shape[0]
        print(self.frames)
        self.label_dimension.setText(str(self.current_index+1)+"/"+str(self.frames)+";  "  +str(self.tiff.shape[2])+"x"+str(self.tiff.shape[1]) + ";  "+str(self.GB)+ " GB;  " + str(self.type))
        self.image_widget.getView().scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.image_widget.timeLine.sigPositionChanged.connect(self.on_time_line_changed)
    
    def cargar_mascara(self, path_mascara):
        self.mascara = tifffile.imread(path_mascara)
        self.mascara = np.transpose(self.mascara, axes=(0, 2, 1))
        #self.mascara = self.mascara[450:,:,:] #SOLO PARA REDUCIR LA MEMORIA
        if self.mascara.shape != self.tiff.shape:
            QMessageBox.warning(self, "Error", "La máscara y el volumen no tienen las mismas dimensiones.")
            return
        self.mascara_exists=True
        # Crear una versión oscurecida de la imagen
        #self.tiff=(self.tiff/255).astype(np.uint8)
        oscurecido_volumen = (self.tiff * 0.5).astype(np.uint8)
        # Aplicar la máscara: mantener la imagen original donde la máscara es blanca (>= 250), y usar la versión oscurecida donde la máscara es negra
        self.volumen_con_mascaras = np.where(self.mascara >= 250, self.tiff, oscurecido_volumen).astype(np.uint8)
        #self.volumen_con_mascaras = oscurecido_volumen

    # def estado_mascara_neuronas(self, state):
    #     if state==0:
    #         self.volumen_a_visualizar=self.tiff
    #         self.checkBox_center_ves.setChecked(False)
    #         self.checkBox_area_ves.setChecked(False)
    #         self.checkBox_mito.setChecked(False)
    #     if state==2:
    #         self.volumen_a_visualizar=self.volumen_con_mascaras

    #     mitocondrias_state = 2 if self.checkBox_mito.isChecked() else 0
    #     self.estado_mascara_mitocondrias(mitocondrias_state)
    
    # def estado_mascara_center_vesicle(self,state):
    #     self.checkBox_neurons.setChecked(True) #para poder visualizar el centro de las vesiculas, primero las neuronas
    #     if self.checkBox_mito.isChecked() and state==0:
    #         self.volumen_a_visualizar=self.volumen_con_mitocondrias
    #         self.checkBox_area_ves.setChecked(False)
    #     if self.checkBox_mito.isChecked() and state==2:
    #         self.checkBox_area_ves.setChecked(True)
    #         self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas
    #     if not self.checkBox_mito.isChecked() and state==2:
    #         self.checkBox_area_ves.setChecked(True)
    #         self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas_sinmito
    #     if not self.checkBox_mito.isChecked() and state==0:
    #         self.volumen_a_visualizar=self.volumen_con_mascaras
    #         self.checkBox_area_ves.setChecked(False)
    #     self.visualizar_imagen(start_index=self.current_index)

    # def estado_mascara_area_vesicle(self,state):
    #     self.checkBox_neurons.setChecked(True)
    #     self.checkBox_center_ves.setChecked(True)
    #     if self.checkBox_mito.isChecked() and state==0:
    #         self.checkBox_center_ves.setChecked(False)
    #         self.volumen_a_visualizar=self.volumen_con_mitocondrias
    #     if self.checkBox_mito.isChecked() and state==2:
    #         self.checkBox_neurons.setChecked(True)
    #         self.checkBox_center_ves.setChecked(True)
    #         self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas
    #     if not self.checkBox_mito.isChecked() and state==2:
    #         self.checkBox_neurons.setChecked(True)
    #         self.checkBox_center_ves.setChecked(True)
    #         self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas_sinmito
    #     if not self.checkBox_mito.isChecked() and state==0:
    #         self.checkBox_center_ves.setChecked(False)
    #         self.volumen_a_visualizar=self.tiff
    #     self.visualizar_imagen(start_index=self.current_index)
    

    # def estado_mascara_mitocondrias(self, state):
    #     if state==0 and self.checkBox_neurons.isChecked():
    #         self.volumen_a_visualizar=self.volumen_con_mascaras
    #         self.checkBox_area_ves.setChecked(False)
    #         self.checkBox_center_ves.setChecked(False)
    #         self.checkBox_mito.setChecked(False)
    #     if state==2 and self.checkBox_neurons.isChecked():
    #         self.volumen_a_visualizar=self.volumen_con_mitocondrias
    #         self.checkBox_area_ves.setChecked(False)
    #         self.checkBox_center_ves.setChecked(False)
    #     if state==2 and not self.checkBox_neurons.isChecked():
    #         self.volumen_a_visualizar=self.volumen_con_mitocondrias_sin_neuronas
    #         self.checkBox_area_ves.setChecked(False)
    #         self.checkBox_center_ves.setChecked(False)
    #     if state==0 and not self.checkBox_neurons.isChecked():
    #         self.checkBox_area_ves.setChecked(False)
    #         self.checkBox_center_ves.setChecked(False)
    #         self.volumen_a_visualizar=self.tiff
    #     self.visualizar_imagen(start_index=self.current_index)

    def opciones_visulalizacion(self):
        if not self.checkBox_neurons.isChecked() and not self.checkBox_mito.isChecked() and not self.checkBox_center_ves.isChecked() and not self.checkBox_area_ves.isChecked():
            self.volumen_a_visualizar=self.tiff
        if self.checkBox_neurons.isChecked() and not self.checkBox_mito.isChecked() and not self.checkBox_center_ves.isChecked() and not self.checkBox_area_ves.isChecked():
            self.volumen_a_visualizar=self.volumen_con_mascaras
        if self.checkBox_neurons.isChecked() and self.checkBox_mito.isChecked() and not self.checkBox_center_ves.isChecked() and not self.checkBox_area_ves.isChecked():
            self.volumen_a_visualizar=self.volumen_con_mitocondrias
        if not self.checkBox_neurons.isChecked() and self.checkBox_mito.isChecked() and not self.checkBox_center_ves.isChecked() and not self.checkBox_area_ves.isChecked():
            self.volumen_a_visualizar=self.volumen_con_mitocondrias_sin_neuronas
        if self.checkBox_area_ves.isChecked() and not self.checkBox_center_ves.isChecked():
            self.checkBox_neurons.setChecked(True)
            if self.checkBox_mito.isChecked():
                self.volumen_a_visualizar=self.volumen_con_areavesiculas
            else:
                self.volumen_a_visualizar=self.volumen_con_areavesiculas_sinmito

        if not self.checkBox_area_ves.isChecked() and self.checkBox_center_ves.isChecked():
            self.checkBox_neurons.setChecked(True)
            if self.checkBox_mito.isChecked():
                self.volumen_a_visualizar=self.volumen_con_centrovesiculas
            else:
                self.volumen_a_visualizar=self.volumen_con_centrovesiculas_sinmito
            #self.checkBox_area_ves.setChecked(True)
        
        if self.checkBox_area_ves.isChecked() and self.checkBox_center_ves.isChecked():
            self.checkBox_neurons.setChecked(True)
            if self.checkBox_mito.isChecked():
                self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas
            else:
                self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas_sinmito
        
        # if self.checkBox_neurons.isChecked() and self.checkBox_mito.isChecked() and self.checkBox_area_ves.isChecked() and not self.checkBox_center_ves.isChecked():
        #     self.volumen_a_visualizar=self.volumen_con_areavesiculas
        
        # if not self.checkBox_neurons.isChecked() and self.checkBox_mito.isChecked() and not self.checkBox_center_ves.isChecked() and not self.checkBox_area_ves.isChecked():
        #     self.volumen_a_visualizar=self.volumen_con_mitocondrias_sin_neuronas
        # if not self.checkBox_neurons.isChecked() and self.checkBox_mito.isChecked() and (self.checkBox_center_ves.isChecked() or self.checkBox_area_ves.isChecked()):
        #     self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas
        #     self.checkBox_area_ves.setChecked(True)
        #     self.checkBox_center_ves.setChecked(True)
        #     self.checkBox_neurons.setChecked(True)
        # if self.checkBox_neurons.isChecked() and not self.checkBox_mito.isChecked() and (self.checkBox_center_ves.isChecked() or self.checkBox_area_ves.isChecked()):
        #     self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas_sinmito
        #     self.checkBox_area_ves.setChecked(True)
        #     self.checkBox_center_ves.setChecked(True)
        # if not self.checkBox_neurons.isChecked() and not self.checkBox_mito.isChecked() and (self.checkBox_center_ves.isChecked() or self.checkBox_area_ves.isChecked()):
        #     self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas_sinmito
        #     self.checkBox_area_ves.setChecked(True)
        #     self.checkBox_center_ves.setChecked(True)
        #     self.checkBox_neurons.setChecked(True)
        self.visualizar_imagen(start_index=self.current_index)
        
            
    def cargar_mascara_mitocondrias(self, path_mascara):
        self.mascara_mitocondrias = (tifffile.imread(path_mascara)).astype(np.uint8)
        self.mascara_mitocondrias = np.transpose(self.mascara_mitocondrias, axes=(0, 2, 1))
        #self.mascara_mitocondrias = self.mascara_mitocondrias[450:,:,:] #ACA SACAR DESPUES
        if self.mascara_mitocondrias.shape != self.tiff.shape:
            QMessageBox.warning(self, "Error", "La máscara de mitocondrias y el volumen no tienen las mismas dimensiones.")
            return
        self.mascara_mitocondrias_exists=True
    
    def colorear_mascara_mitocondrias(self, volumen, mask, RGB, un_slide=False):  
        # Definir el color rojo translúcido
        alpha = 0.2
        red_color = np.array([253, 60, 20], dtype=np.uint8)  # Color rojo
        print("rojo:",red_color.shape)

        if not RGB:
            # Convertir la imagen a color
            volumen = np.stack([volumen] * 3, axis=-1)

        print("volumen:",volumen.shape)
        print("mask:",mask.shape)
        # Preparar el volumen con máscaras
        volumen_con_mitocondrias = np.copy(volumen)
        # Crear la superposición roja
        red_overlay = np.zeros_like(volumen)
        if un_slide:
            red_overlay[(mask >= 250)  & (self.tiff[self.current_index,:,:] <=205)] = red_color
        else:
            red_overlay[(mask >= 250)  & (self.tiff <=205)] = red_color
        #volumen_nosaturado=volumen
        #volumen_nosaturado[volumen>=205]=205
        # Aplicar la transparencia sobre toda la imagen
        #volumen_con_mitocondrias = (alpha * red_overlay + (1 - alpha) * volumen_color).astype(np.uint8)
        volumen_con_mitocondrias = (alpha * red_overlay + volumen).astype(np.uint8)
        return volumen_con_mitocondrias
    
    def colorear_areavesiculas(self, volumen, mask, RGB):  
        # Definir el color rojo translúcido
        alpha = 0.2
        blue_color = np.array([20, 60, 253], dtype=np.uint8)  # Color azul

       

        if not RGB:
            # Convertir la imagen a color
            volumen = np.stack([volumen] * 3, axis=-1)

        # Preparar el volumen con máscaras
        volumen_con_areavesiculas = np.copy(volumen)
        # Crear la superposición roja
        print("shape volumen", volumen.shape)
        print("shape mask: ", mask.shape)
        blue_overlay = np.zeros_like(volumen)
        blue_overlay[(mask >= 250) & (self.tiff <=205)] = blue_color
        #volumen_nosaturado=volumen
        #volumen_nosaturado[volumen>=205]=205
        # Aplicar la transparencia sobre toda la imagen
        #volumen_con_mitocondrias = (alpha * red_overlay + (1 - alpha) * volumen_color).astype(np.uint8)
        volumen_con_areavesiculas = (alpha * blue_overlay + volumen).astype(np.uint8)
        #volumen_con_areavesiculas = volumen_con_areavesiculas[..., 0]
        return volumen_con_areavesiculas
    
    def colorear_centrovesiculas(self, volumen, mask, RGB):  
        # Definir el color rojo translúcido
        alpha = 0.9
        green_color = np.array([60, 255, 0], dtype=np.uint8)  # Color amarillo

        if not RGB:
            # Convertir la imagen a color
            volumen = np.stack([volumen] * 3, axis=-1)
    

        # Preparar el volumen con máscaras
        volumen_con_centrovesiculas = np.copy(volumen)

        yellow_overlay = np.copy(volumen)
        yellow_overlay[mask >= 250] = green_color
        
        volumen_con_centrovesiculas = (yellow_overlay).astype(np.uint8)
        #volumen_con_centrovesiculas = (alpha * yellow_overlay + volumen).astype(np.uint8)
        #volumen_con_centrovesiculas=volumen_con_centrovesiculas[...,0,0]
        return volumen_con_centrovesiculas


    def visualizar_imagen(self, start_index):
        self.image_widget.getView().setAspectLocked(True)
        self.image_widget.getView().autoRange(padding=0.0)
        self.image_widget.timeLine.setPen((255,255,0,200))
        self.image_widget.setImage(self.volumen_a_visualizar)
        self.image_widget.setCurrentIndex(start_index)
        
    def on_time_line_changed(self):
        self.current_index = int(self.image_widget.currentIndex)
        #print(f"Current frame index: {self.current_index}")
        self.label_dimension.setText(str(self.current_index+1)+"/"+str(self.frames)+";  "  +str(self.tiff.shape[2])+"x"+str(self.tiff.shape[1]) + ";  "+str(self.GB)+ " GB;  " + str(self.type))
    

    def on_mouse_moved(self, pos):
        mouse_point = self.image_widget.getView().mapSceneToView(pos)
        self.x, self.y = int(mouse_point.x()), int(mouse_point.y())

        if 0 <= self.x < self.tiff.shape[1] and 0 <= self.y < self.tiff.shape[2]:
            self.label_coordenadas.setText(f"x= {self.x}, y= {self.y}, z= {self.current_index}")
            self.label_value.setText(f"value= {self.tiff[self.current_index, self.x, self.y]}")
            if self.mascara_etiquetada_mitochondria_exists:
                label=self.mascara_etiquetada_mitochondria[self.current_index,self.x,self.y]
                if label!=0:
                    self.label_mitocondria.setText(f"label mitochondria: {label}")
                else:
                    self.label_mitocondria.setText(f"label mitochondria:")
            if self.mascara_area_vesicles_etiquetada_exists:
                label2=self.mascara_area_vesicles_etiquetada[self.current_index,self.x,self.y]
                if label2!=0:
                    self.label_vesicula.setText(f"label vesicle: {label2}")
                else:
                    self.label_vesicula.setText(f"label vesicle:")

        else:
            self.label_coordenadas.setText("")
            self.label_value.setText("")
    
    def create_new_project(self):
        # Pedir nombre del proyecto
        self.project_name, ok = QInputDialog.getText(self, 'Nuevo Proyecto', 'Ingrese el nombre del proyecto:')
        if not ok or not self.project_name:
            return #usuario cancelo la entrada del nombre
        
        # Pedir tamaño de píxel
        pixel_size, ok = QInputDialog.getDouble(self, 'Tamaño de Píxel', 'Ingrese el tamaño del píxel (nm):', decimals=2)
        if not ok:
            return  # El usuario canceló la entrada del tamaño de píxel
        self.pixel_size = pixel_size
        
        # Pedir escala Z
        z_scale, ok = QInputDialog.getDouble(self, 'Escala Z', 'Ingrese la escala Z:', decimals=2)
        if not ok:
            return  # El usuario canceló la entrada de la escala Z
        self.z_scale = z_scale
        
        # Seleccionar la carpeta donde se creará el proyecto
        project_dir = QFileDialog.getExistingDirectory(self, 'Seleccione la carpeta donde se creará el proyecto')
        if not project_dir:
            return  # El usuario canceló la selección de la carpeta
        
        project_dir = os.path.join(project_dir, self.project_name)
        
        try:
            os.makedirs(project_dir)
        except FileExistsError:
            QMessageBox.warning(self, "Error", "El proyecto ya existe.")
            return
        
        # Seleccionar el primer archivo (volumen a analizar)
        volume_file, _ = QFileDialog.getOpenFileName(self, "Seleccione el volumen a analizar", "", "Archivos (*.tif *.tiff)")
        if not volume_file:
            QMessageBox.warning(self, "Error", "Debe seleccionar un archivo de volumen.")
            return
        
        # Seleccionar el segundo archivo (máscara)
        mask_file, _ = QFileDialog.getOpenFileName(self, "Seleccione la máscara de neuronas de interes, en caso de tenerla", "", "Archivos (*.tif *.tiff)")
        if not mask_file:
            QMessageBox.warning(self, "Advertencia!", "No se ha seleccionado un archivo de mascara")
            #pass
        else:
            mask_dest = os.path.join(project_dir, 'mascara_neuronas' + os.path.splitext(mask_file)[1])
            shutil.copy2(mask_file, mask_dest)
        
        # Copiar los archivos a la nueva carpeta

        volume_dest = os.path.join(project_dir, 'volumen' + os.path.splitext(volume_file)[1])
        
        shutil.copy2(volume_file, volume_dest)
        
        # Guardar los detalles del proyecto en un archivo de texto
        project_details_path = os.path.join(project_dir, f'{self.project_name}.txt')
        with open(project_details_path, 'w') as f:
            f.write(f'#Project Name: {self.project_name}\n')
            f.write(f'# Pixel Size (nm): {self.pixel_size}\n')
            f.write(f'# Z Scale: {self.z_scale}\n')
        
        QMessageBox.information(self, "Éxito", f"El proyecto '{self.project_name}' ha sido creado y los archivos han sido copiados.")
        self.analizar_carpeta(project_dir)
        self.path_carpeta=project_dir
        parent_dir = os.path.dirname(project_dir)
        self.parent_path=parent_dir
        self.setWindowTitle(self.project_name)

    def open_project(self):
        # Seleccionar la carpeta del proyecto existente
        project_dir = QFileDialog.getExistingDirectory(self, 'Seleccione la carpeta del proyecto existente')
        if not project_dir:
            return None  # El usuario canceló la selección de la carpeta

        # Verificar la existencia de archivos clave en la carpeta
        self.project_name = os.path.basename(project_dir)
        volumen_path = None

        for archivo in os.listdir(project_dir):
            archivo_path = os.path.join(project_dir, archivo)
            
            if archivo.lower() == 'volumen.tif' or archivo.lower() == 'volumen.tiff':
                volumen_path = archivo_path

        if not volumen_path:
            QMessageBox.warning(self, "Error", "No se encontró el archivo 'volumen.tif' o 'volumen.tiff' en la carpeta seleccionada.")
            return None
        
        # Leer el archivo de configuración
        config_file = os.path.join(project_dir, f'{self.project_name}.txt')
        if os.path.exists(config_file):
            with open(config_file, 'r') as file:
                for line in file:
                    if line.startswith('# Pixel Size (nm):'):
                        self.pixel_size = float(line.split(':')[1].strip())
                    elif line.startswith('# Z Scale:'):
                        self.z_scale = float(line.split(':')[1].strip())
        else:
            QMessageBox.warning(self, "Advertencia!", "No se encontró el archivo de configuración del proyecto.")
        
        # carpeta en donde esta la carpeta del proyecto (aca adentro tambien estan los archivos py y pesos de los modelos)
        parent_dir = os.path.dirname(project_dir)
        self.analizar_carpeta(project_dir)
        self.path_carpeta=project_dir #path a la carpeta del proyecto
        self.parent_path=parent_dir #path a la carpeta donde estan todos los codigos + la carpeta del proyecto
        self.setWindowTitle(self.project_name)

    def analizar_carpeta(self, project_dir):
    # Inicializamos los paths como None
        path_volumen = None
        path_mascara_neuronas = None
        path_mascara_mitocondrias = None
        self.mascara_exists=False
        self.mascara_mitocondrias_exists=False
        self.multiplied_exists=False
        self.mascara_etiquetada_mitochondria_exists=False

        self.mascara_center_vesicle_exists=False
        self.mascara_area_vesicles_exists=False
        self.mascara_center_vesicles_etiquetada_exists=False
        self.mascara_area_vesicles_etiquetada_exists=False

        path_center_vesicle = None
        path_area_vesicles=None
        path_center_vesicles_etiquetada=None
        path_area_vesicles_etiquetada=None

        

        # Recorremos los archivos de la carpeta
        for archivo in os.listdir(project_dir):
            archivo_path = os.path.join(project_dir, archivo)
            if archivo.lower() == 'volumen.tif' or archivo.lower() == 'volumen':
                path_volumen = archivo_path
                self.cargar_imagen(path_volumen)
        for archivo in os.listdir(project_dir):
            archivo_path = os.path.join(project_dir, archivo)
            if archivo.lower() == 'mascara_neuronas.tif' or archivo.lower() == 'mascara_neuronas':
                path_mascara_neuronas = archivo_path
                self.checkBox_neurons.setChecked(True)
                self.cargar_mascara(path_mascara_neuronas)
                self.volumen_a_visualizar=self.volumen_con_mascaras
        for archivo in os.listdir(project_dir):
            archivo_path = os.path.join(project_dir, archivo)
            # Verificamos si el archivo es mascaras_mitocondrias
            if archivo.lower() == 'mascara_mitocondrias.tif':
                path_mascara_mitocondrias = archivo_path
                self.checkBox_mito.setChecked(True)
                self.cargar_mascara_mitocondrias(path_mascara_mitocondrias)
                self.volumen_con_mitocondrias=self.colorear_mascara_mitocondrias(self.volumen_con_mascaras, self.mascara_mitocondrias,RGB=False)
                print("volumen con mitocondrias shape: ",self.volumen_con_mitocondrias.shape)
                self.volumen_con_mitocondrias_sin_neuronas=self.colorear_mascara_mitocondrias(self.tiff,self.mascara_mitocondrias,RGB=False)
                self.volumen_a_visualizar=self.volumen_con_mitocondrias
        
        for archivo in os.listdir(project_dir):
            archivo_path = os.path.join(project_dir, archivo)
            # Verificamos si el archivo es mascaras_mitocondrias_etiquetada
            if archivo.lower() == 'mascara_mitocondrias_etiquetada.tif':
                path_mascara_mitocondrias_etiquetada = archivo_path
                self.mascara_etiquetada_mitochondria_exists=True
                self.mascara_etiquetada_mitochondria = tifffile.imread(path_mascara_mitocondrias_etiquetada)
                self.mascara_etiquetada_mitochondria = np.transpose(self.mascara_etiquetada_mitochondria, axes=(0, 2, 1))

        for archivo in os.listdir(project_dir):
            archivo_path = os.path.join(project_dir, archivo)
            # Verificamos si el archivo es mascaras_vesiculas
            if archivo.lower() == 'mascara_area_vesicles.tif':
                path_area_vesicles = archivo_path
                self.mascara_area_vesicles_exists=True
                self.checkBox_area_ves.setChecked(True)
                self.mascara_area_vesicles=tifffile.imread(path_area_vesicles)
                self.mascara_area_vesicles = np.transpose(self.mascara_area_vesicles, axes=(0, 2, 1))
                print('Hay mascaras de area vesiculas')
                # self.volumen_con_areavesiculas=self.colorear_areavesiculas(self.volumen_con_mitocondrias,self.mascara_area_vesicles,RGB=True)
                # self.volumen_a_visualizar=self.volumen_con_areavesiculas
                # self.volumen_con_areaycentrovesiculas=self.colorear_centrovesiculas(self.volumen_con_areavesiculas,self.mascara_center_vesicles,RGB=True)
                # self.volumen_con_areavesiculas_sinmito=self.colorear_areavesiculas(self.volumen_con_mascaras,self.mascara_area_vesicles,RGB=False)
                # self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas
        
        for archivo in os.listdir(project_dir):
            archivo_path = os.path.join(project_dir, archivo)
            # Verificamos si el archivo es mascaras_vesiculas
            if archivo.lower() == 'mascara_center_vesicles.tif':
                path_center_vesicle = archivo_path
                self.mascara_center_vesicle_exists=True
                self.checkBox_center_ves.setChecked(True)
                self.mascara_center_vesicles=tifffile.imread(path_center_vesicle)
                self.mascara_center_vesicles = np.transpose(self.mascara_center_vesicles, axes=(0, 2, 1))
                print('Hay mascaras de vesiculas')
                self.volumen_con_centrovesiculas=self.colorear_centrovesiculas(self.volumen_con_mitocondrias,self.mascara_center_vesicles,RGB=True)
                self.volumen_con_areavesiculas=self.colorear_areavesiculas(self.volumen_con_mitocondrias,self.mascara_area_vesicles,RGB=True)
                self.volumen_con_areaycentrovesiculas=self.colorear_centrovesiculas(self.volumen_con_areavesiculas,self.mascara_center_vesicles,RGB=True)
                self.volumen_con_areavesiculas_sinmito=self.colorear_areavesiculas(self.volumen_con_mascaras,self.mascara_area_vesicles,RGB=False)
                self.volumen_con_centrovesiculas_sinmito=self.colorear_centrovesiculas(self.volumen_con_mascaras, self.mascara_center_vesicles, RGB=False )
                self.volumen_con_areaycentrovesiculas_sinmito=self.colorear_centrovesiculas(self.volumen_con_areavesiculas_sinmito,self.mascara_center_vesicles,RGB=True)
                self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas
        
        
        
        for archivo in os.listdir(project_dir):
            archivo_path = os.path.join(project_dir, archivo)
            # Verificamos si el archivo es mascaras_vesiculas
            if archivo.lower() == 'mascara_area_vesicles_etiquetada.tif':
                path_area_vesicles_etiquetada = archivo_path
                self.mascara_area_vesicles_etiquetada_exists=True
                self.mascara_area_vesicles_etiquetada=tifffile.imread(path_area_vesicles_etiquetada)
                self.mascara_area_vesicles_etiquetada = np.transpose(self.mascara_area_vesicles_etiquetada, axes=(0, 2, 1))
                print('Hay mascaras de area vesiculas etiquetadas')
        
        for archivo in os.listdir(project_dir):
            archivo_path = os.path.join(project_dir, archivo)
            # Verificamos si el archivo es mascaras_vesiculas
            if archivo.lower() == 'mascara_center_vesicles_etiquetada.tif':
                path_center_vesicles_etiquetada = archivo_path
                self.mascara_center_vesicles_etiquetada_exists=True
                self.mascara_center_vesicles_etiquetada=tifffile.imread(path_center_vesicles_etiquetada)
                self.mascara_center_vesicles_etiquetada = np.transpose(self.mascara_center_vesicles_etiquetada, axes=(0, 2, 1))
                print('Hay mascaras de vesiculas centro etiquetadas')
                
        self.visualizar_imagen(start_index=self.current_index)
    
    def save_project(self):
        path_mitocondria= os.path.join(self.path_carpeta, 'mascara_mitocondrias.tif')
        tifffile.imsave(path_mitocondria, np.transpose(self.mascara_mitocondrias, axes=(0, 2, 1)))
        path_mitocondria_etiquetada= os.path.join(self.path_carpeta, 'mascara_mitocondrias_etiquetada.tif')
        tifffile.imsave(path_mitocondria_etiquetada, np.transpose(self.mascara_etiquetada_mitochondria, axes=(0, 2, 1)))
        path_center_vesicle= os.path.join(self.path_carpeta, 'mascara_center_vesicles.tif')
        tifffile.imsave(path_center_vesicle, np.transpose(self.mascara_center_vesicles, axes=(0,2,1)))
        path_area_vesicles= os.path.join(self.path_carpeta, 'mascara_area_vesicles.tif')
        tifffile.imsave(path_area_vesicles, np.transpose(self.mascara_area_vesicles, axes=(0,2,1)))
        path_center_vesicles_etiquetada= os.path.join(self.path_carpeta, 'mascara_center_vesicles_etiquetada.tif')
        tifffile.imsave(path_center_vesicles_etiquetada,np.transpose(self.mascara_center_vesicles_etiquetada, axes=(0,2,1)))
        path_area_vesicles_etiquetada= os.path.join(self.path_carpeta, 'mascara_area_vesicles_etiquetada.tif')
        tifffile.imsave(path_area_vesicles_etiquetada, np.transpose(self.mascara_area_vesicles_etiquetada, axes=(0,2,1)))
 
    
    def cargar_multiplied(self):
        for archivo in os.listdir(self.path_carpeta):
            archivo_path = os.path.join(self.path_carpeta, archivo)
            if archivo.lower() == 'multiplied.tif':
                multiplied_path = archivo_path
                self.multiplied = tifffile.imread(multiplied_path)
                self.multiplied = np.transpose(self.multiplied, axes=(0, 2, 1))
                self.multiplied_exists=True
        if not self.mascara_exists:
            QMessageBox.warning(self, "Error", "Debe tener una máscara de neuronas para predecir mitocondrias y vesiculas.")
            return None
        if not self.multiplied_exists or (self.multiplied.shape != self.tiff.shape):
            self.multiplied=multiply_mask(self.tiff, self.mascara, self.path_carpeta, True) #se crea la multiplied y se guarda
    
    def predict_mitochondria (self):
        self.cargar_multiplied()
        self.mascara_mitocondrias=predict_mito(self.multiplied, self.parent_path, self.path_carpeta) #se predice y se guarda mascara mitocondrias
        self.checkBox_mito.setChecked(True)
        self.mascara_mitocondrias_exists=True
        self.volumen_con_mitocondrias=self.colorear_mascara_mitocondrias(self.volumen_con_mascaras, self.mascara_mitocondrias,RGB=False)
        self.volumen_con_mitocondrias_sin_neuronas=self.colorear_mascara_mitocondrias(self.tiff,self.mascara_mitocondrias,RGB=False)
        self.volumen_a_visualizar=self.volumen_con_mitocondrias
        self.visualizar_imagen(start_index=self.current_index)
    
    def chequeo_vesiculas_mitocondrias(self):
        coincidencias = (self.mascara_center_vesicles == 255) & (self.mascara_mitocondrias == 255)
        self.mascara_center_vesicles[coincidencias] = 0
        return self.mascara_center_vesicles


    def elegir_metodo_prediccion_ves(self):
        if self.mascara_exists:
            # Crear un cuadro de diálogo
            dialog = QDialog(self)
            dialog.setWindowTitle("Seleccionar método de predicción")
            dialog.setGeometry(150, 150, 300, 150)

            # Crear layout vertical
            layout = QVBoxLayout(dialog)

            # Crear radio buttons para opciones
            option1 = QRadioButton("Opción 1 - Prediccion de vesiculas utilizando bin1 - Accuracy: xx%", dialog)
            option2 = QRadioButton("Opción 2 - Prediccion de vesiculas utilizando bin5 - Accuracy: xx%", dialog)

            # Seleccionar por defecto la primera opción
            option1.setChecked(True)

            # Añadir radio buttons al layout
            layout.addWidget(option1)
            layout.addWidget(option2)

            # Crear botón para confirmar selección
            button = QPushButton("Confirmar", dialog)
            button.clicked.connect(dialog.accept)  # Cierra el diálogo con QDialog.Accepted

            # Añadir botón al layout
            layout.addWidget(button)

            # Mostrar el cuadro de diálogo
            result = dialog.exec_()

            # Dependiendo de la selección del usuario, mostrar un mensaje o realizar alguna acción
            if result == QDialog.Accepted:
                if option1.isChecked():
                    QMessageBox.information(self, "Selección", "Opción 1 seleccionada: Ejecutando prediccion con bin1")
                    self.predict_ves_bin1()
                else:
                    QMessageBox.information(self, "Selección", "Opción 2 seleccionada: Ejecutando prediccion con bin5")
                    self.predict_ves_bin5()
        else:
            QMessageBox.warning(self, "Error", "No se puede predecir las vesiculas sin antes tener una mascara de neuronas. Primero realiza la prediccion de neuronas de interes")
            print("No se puede predecir las vesiculas sin antes tener una mascara de neuronas. Primero realiza la prediccion de neuronas de interes")
            return
    
    def predict_ves_bin1(self):
        
        #con bin1 de pedir el path para el bin1 (sin moverlo a la carpeta) y despues resizear el resultado para que quede ok para bin 5
        pixel_size_bin1, ok = QInputDialog.getDouble(self, 'Tamaño de Píxel bin 1', 'Ingrese el tamaño del píxel (nm):', decimals=2)
        if not ok:
            return  # El usuario canceló la entrada del tamaño de píxel
        
        ruta_bin1, ok = QInputDialog.getText(self, 'Ruta de acceso bin 1', 'Ingrese la ruta completa al archivo bin 1, incluyendo el nombre del archivo: ')
        if not ok:
            return

        project_details_bin1_path = os.path.join(self.path_carpeta, f'{self.project_name}_bin1.txt')
        with open(project_details_bin1_path, 'w') as f:
            f.write(f'# Project Name: {self.project_name}_bin1\n')
            f.write(f'# Pixel Size bin 1 (nm): {pixel_size_bin1}\n')
            f.write(f'# Ruta de acceso + nombre: {ruta_bin1}')
            
        # Leer el archivo de configuración
        config_file_bin1 = os.path.join(self.path_carpeta, f'{self.project_name}_bin1.txt')
        if os.path.exists(config_file_bin1):
            with open(config_file_bin1, 'r') as file:
                for line in file:
                    if line.startswith('# Pixel Size bin 1 (nm):'):
                        self.pixel_size = float(line.split(':')[1].strip())
                    elif line.startswith('# Ruta de acceso + nombre:'):
                        # Extraer la parte después de 'r''
                        inicio = line.find("r'") + 2
                        # Obtener la ruta sin el prefijo y las comillas finales
                        self.ruta_bin1 = line[inicio:].strip().strip("'")
        print(self.ruta_bin1)
        self.cargar_multiplied()
        self.bin1=tifffile.imread(self.ruta_bin1)
        self.bin1=np.transpose(self.bin1, axes=(0,2,1))
        #self.bin1=self.bin1[100:,:,:]
        print(self.bin1.shape)

        tamanobin1=self.bin1[0].shape
        tamanobin5=self.mascara[0].shape
        
        multiplied=False
        if multiplied==False: #siempre exepto este caso, despues borro esto
            #resizear mask bin5 a bin 1
            self.mascara_bin1 = np.zeros((self.bin1.shape[0], self.bin1.shape[1], self.bin1.shape[2]), dtype=np.uint8)

            # Redimensionar cada "corte" del volumen
            for i in range(self.bin1.shape[0]):
                self.mascara_bin1[i] = cv2.resize(self.mascara[i],((tamanobin1[1],tamanobin1[0])), interpolation=cv2.INTER_NEAREST_EXACT)
            
            self.multiplied_bin1=multiply_mask(self.bin1, self.mascara_bin1, self.path_carpeta, False) #se crea pero no se guarda
        else:
            self.multiplied_bin1=self.bin1  
        #probando poniendo directo la mascara multiplied bin1, despues va a haber que pedir que ingrese el path al volumen bin1 y generar esta mascara
        #self.multiplied_bin1=tifffile.imread(r'\Users\ulabceriani\Documents\ITBA\ITBA_ZT2\multiplied_bin1_zt2_4slices.tif')
        self.mascara_area_vesicles_etiquetada, self.mascara_center_vesicles = predict_vesicles(self.multiplied_bin1, self.parent_path, self.path_carpeta, tamanobin5, tamanobin1) #se predice y se guarda mascara vesiculas (probability map color semi final y centers final)
        self.chequeo_vesiculas_mitocondrias()
        self.mascara_center_vesicle_exists=True
        path_center_vesicle= os.path.join(self.path_carpeta, 'mascara_center_vesicles.tif')
        tifffile.imsave(path_center_vesicle, np.transpose(self.mascara_center_vesicles, axes=(0,2,1)))
        print(self.mascara_center_vesicles.shape)
        self.label_ves()
        self.mascara_area_vesicles_exists=True
        self.checkBox_area_ves.setChecked(True)
        self.checkBox_center_ves.setChecked(True)
        self.volumen_con_centrovesiculas=self.colorear_centrovesiculas(self.volumen_con_mitocondrias,self.mascara_center_vesicles,RGB=True)
        self.volumen_con_areavesiculas=self.colorear_areavesiculas(self.volumen_con_mitocondrias,self.mascara_area_vesicles,RGB=True)
        self.volumen_con_areaycentrovesiculas=self.colorear_centrovesiculas(self.volumen_con_areavesiculas,self.mascara_center_vesicles,RGB=True)
        self.volumen_con_areavesiculas_sinmito=self.colorear_areavesiculas(self.volumen_con_mascaras,self.mascara_area_vesicles,RGB=False)
        self.volumen_con_centrovesiculas_sinmito=self.colorear_centrovesiculas(self.volumen_con_mascaras, self.mascara_center_vesicles, RGB=False )
        self.volumen_con_areaycentrovesiculas_sinmito=self.colorear_centrovesiculas(self.volumen_con_areavesiculas_sinmito,self.mascara_center_vesicles,RGB=True)
        self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas
        self.visualizar_imagen(start_index=self.current_index)

    def predict_ves_bin5(self):
        self.cargar_multiplied()
        tamanobin5=self.mascara[0].shape
        self.mascara_area_vesicles_etiquetada, self.mascara_center_vesicles = predict_vesicles_bin5(self.multiplied, self.parent_path, self.path_carpeta, tamanobin5) #se predice y se guarda mascara vesiculas (probability map color semi final y centers final)
        self.chequeo_vesiculas_mitocondrias()
        self.mascara_center_vesicle_exists=True
        path_center_vesicle= os.path.join(self.path_carpeta, 'mascara_center_vesicles.tif')
        tifffile.imsave(path_center_vesicle, np.transpose(self.mascara_center_vesicles, axes=(0,2,1)))
        print(self.mascara_center_vesicles.shape)
        self.label_ves()
        self.mascara_area_vesicles_exists=True
        self.checkBox_area_ves.setChecked(True)
        self.checkBox_center_ves.setChecked(True)
        self.volumen_con_centrovesiculas=self.colorear_centrovesiculas(self.volumen_con_mitocondrias,self.mascara_center_vesicles,RGB=True)
        self.volumen_con_areavesiculas=self.colorear_areavesiculas(self.volumen_con_mitocondrias,self.mascara_area_vesicles,RGB=True)
        self.volumen_con_areaycentrovesiculas=self.colorear_centrovesiculas(self.volumen_con_areavesiculas,self.mascara_center_vesicles,RGB=True)
        self.volumen_con_areavesiculas_sinmito=self.colorear_areavesiculas(self.volumen_con_mascaras,self.mascara_area_vesicles,RGB=False)
        self.volumen_con_centrovesiculas_sinmito=self.colorear_centrovesiculas(self.volumen_con_mascaras, self.mascara_center_vesicles, RGB=False )
        self.volumen_con_areaycentrovesiculas_sinmito=self.colorear_centrovesiculas(self.volumen_con_areavesiculas_sinmito,self.mascara_center_vesicles,RGB=True)
        self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas
        self.visualizar_imagen(start_index=self.current_index)



    def label_ves(self):
        if (self.mascara_center_vesicle_exists):
            self.mascara_center_vesicles_etiquetada, self.mascara_area_vesicles_etiquetada, self.mascara_area_vesicles, self.cantidad_vesiculas= etiquetado_vesiculas_unicas(self.mascara_center_vesicles,self.mascara_area_vesicles_etiquetada, self.path_carpeta)
            #write self.cantidad_vesiculas
            project_details_path = os.path.join(self.path_carpeta, f'{self.project_name}.txt')
            with open(project_details_path, 'a') as f:
                f.write(f'#Cantidad de vesiculas: {self.cantidad_vesiculas}\n')
                
                
            self.mascara_center_vesicles_etiquetada_exists=True
            self.mascara_area_vesicles_etiquetada_exists=True
            path_area_vesicles= os.path.join(self.path_carpeta, 'mascara_area_vesicles.tif')
            tifffile.imsave(path_area_vesicles, np.transpose(self.mascara_area_vesicles, axes=(0,2,1)))
            path_center_vesicles_etiquetada= os.path.join(self.path_carpeta, 'mascara_center_vesicles_etiquetada.tif')
            tifffile.imsave(path_center_vesicles_etiquetada,np.transpose(self.mascara_center_vesicles_etiquetada, axes=(0,2,1)))
            path_area_vesicles_etiquetada= os.path.join(self.path_carpeta, 'mascara_area_vesicles_etiquetada.tif')
            tifffile.imsave(path_area_vesicles_etiquetada, np.transpose(self.mascara_area_vesicles_etiquetada, axes=(0,2,1)))
        else:
            QMessageBox.warning(self, "Error", "No se puede etiquetar las vesiculas sin antes tener una mascara de vesiculas. Primero realiza la prediccion de vesiculas")
            print("No se puede etiquetar las vesiculas sin antes tener una mascara de vesiculas. Primero realiza la prediccion de vesiculas")
            return

    def predict_neurons(self):
        self.mascara,self.mascara_mitocondrias=predict_neurons(self.tiff, self.parent_path, self.path_carpeta)
        self.checkBox_mito.setChecked(True)
        self.checkBox_neurons.setChecked(True)
        self.mascara_exists=True
        self.mascara_mitocondrias_exists=True
        # Crear una versión oscurecida de la imagen
        oscurecido_volumen = (self.tiff * 0.5).astype(np.uint8)
        # Aplicar la máscara: mantener la imagen original donde la máscara es blanca (>= 250), y usar la versión oscurecida donde la máscara es negra
        self.volumen_con_mascaras = np.where(self.mascara >= 250, self.tiff, oscurecido_volumen).astype(np.uint8)
        self.volumen_con_mitocondrias=self.colorear_mascara_mitocondrias(self.volumen_con_mascaras, self.mascara_mitocondrias,RGB=False)
        self.volumen_con_mitocondrias_sin_neuronas=self.colorear_mascara_mitocondrias(self.tiff,self.mascara_mitocondrias,RGB=False)
        self.volumen_a_visualizar=self.volumen_con_mitocondrias
        self.visualizar_imagen(start_index=self.current_index)
    

    def label_mitochondria(self):
        self.mascara_etiquetada_mitochondria = mitochondria_labeling (self.mascara_mitocondrias)
        self.mascara_etiquetada_mitochondria_exists=True
        path_mitocondria_etiquetada= os.path.join(self.path_carpeta, 'mascara_mitocondrias_etiquetada.tif')
        tifffile.imsave(path_mitocondria_etiquetada, np.transpose(self.mascara_etiquetada_mitochondria, axes=(0, 2, 1)))


    def add_mitochondria(self):
        self.checkBox_neurons.setChecked(True)
        self.checkBox_mito.setChecked(True)
        self.checkBox_area_ves.setChecked(False)
        self.checkBox_center_ves.setChecked(False)
        if self.mouse_click_connection is None:
            self.mouse_click_connection = self.image_widget.scene.sigMouseClicked.connect(self.on_mouse_clicked)
        self.label_tool.setText('You are adding mitochondria')

    def delete_mitochondria(self):
        if self.mouse_click_connection is None:
                    self.mouse_click_connection = self.image_widget.scene.sigMouseClicked.connect(self.on_mouse_clicked_delete)
        self.label_tool.setText('You are deleting mitochondria')
    
    def delete_vesicle(self):
        if self.mouse_click_connection is None:
                    self.mouse_click_connection = self.image_widget.scene.sigMouseClicked.connect(self.on_mouse_clicked_delete_vesicles)
        self.label_tool.setText('You are deleting vesicle')
    
    def finish_click(self):
        # Desconectar el evento de clic del mouse de la función
        if self.mouse_click_connection is not None:
            if self.label_tool.text()=='You are adding mitochondria':
                self.image_widget.scene.sigMouseClicked.disconnect(self.on_mouse_clicked)
            if self.label_tool.text()=='You are adding vesicle':
                self.image_widget.scene.sigMouseClicked.disconnect(self.on_mouse_clicked_vesicles)
            self.mouse_click_connection = None
        self.label_tool.setText('')
    
    def finish_delete(self):
        # Desconectar el evento de clic del mouse de la función
        if self.mouse_click_connection is not None:
            if self.label_tool.text()=='You are deleting mitochondria':
                self.image_widget.scene.sigMouseClicked.disconnect(self.on_mouse_clicked_delete)
            if self.label_tool.text()=='You are deleting vesicle':
                self.image_widget.scene.sigMouseClicked.disconnect(self.on_mouse_clicked_delete_vesicles)
            self.mouse_click_connection = None
            
        self.label_tool.setText('')
    
    def add_vesicles(self):
        self.checkBox_neurons.setChecked(True)
        self.checkBox_mito.setChecked(False)
        self.checkBox_area_ves.setChecked(True)
        self.checkBox_center_ves.setChecked(True)
        
        if self.mouse_click_connection is None:
                    self.mouse_click_connection = self.image_widget.scene.sigMouseClicked.connect(self.on_mouse_clicked_vesicles)
        self.label_tool.setText('You are adding vesicle')

    
    def on_mouse_clicked(self, event):
        self.mascara_mitocondrias_undo=(self.mascara_mitocondrias).copy()
        self.volumen_con_mitocondrias_undo=(self.volumen_con_mitocondrias).copy()
        self.volumen_con_mitocondrias_sin_neuronas_undo=(self.volumen_con_mitocondrias_sin_neuronas).copy()
        # Get the position of the mouse click in the image
        mouse_point = self.image_widget.getView().mapSceneToView(event.scenePos())
        self.x, self.y = int(mouse_point.x()), int(mouse_point.y())
        # Print the coordinates
        print(f"Mouse clicked at: x={self.x}, y={self.y}")
        new_mask=region_growing(self.tiff[self.current_index,:,:],self.mascara_mitocondrias[self.current_index,:,:],self.mascara[self.current_index,:,:],[(self.x,self.y)],30)
        self.mascara_mitocondrias[self.current_index,:,:]=new_mask
        self.colorear_un_slide=True
        self.volumen_con_mitocondrias[self.current_index,:,:]=self.colorear_mascara_mitocondrias(self.volumen_con_mascaras[self.current_index,:,:],new_mask,RGB=False, un_slide=True)
        self.volumen_con_mitocondrias_sin_neuronas[self.current_index,:,:]=self.colorear_mascara_mitocondrias(self.tiff[self.current_index,:,:],new_mask,RGB=False, un_slide=True)
        self.visualizar_imagen(start_index=self.current_index)
    
    def on_mouse_clicked_delete(self, event):
        self.mascara_mitocondrias_undo=(self.mascara_mitocondrias).copy()
        self.volumen_con_mitocondrias_undo=(self.volumen_con_mitocondrias).copy()
        self.volumen_con_mitocondrias_sin_neuronas_undo=(self.volumen_con_mitocondrias_sin_neuronas).copy()
        # Get the position of the mouse click in the image
        mouse_point = self.image_widget.getView().mapSceneToView(event.scenePos())
        self.x, self.y = int(mouse_point.x()), int(mouse_point.y())
        # Print the coordinates
        print(f"Mouse clicked at: x={self.x}, y={self.y}")
        new_mask=region_growing_delete(self.tiff[self.current_index,:,:],self.mascara_mitocondrias[self.current_index,:,:],self.mascara[self.current_index,:,:],[(self.x,self.y)])
        self.mascara_mitocondrias[self.current_index,:,:]=new_mask
        self.volumen_con_mitocondrias[self.current_index,:,:]=self.colorear_mascara_mitocondrias(self.volumen_con_mascaras[self.current_index,:,:],new_mask,RGB=False, un_slide=True)
        self.volumen_con_mitocondrias_sin_neuronas[self.current_index,:,:]=self.colorear_mascara_mitocondrias(self.tiff[self.current_index,:,:],new_mask,RGB=False, un_slide=True)
        self.visualizar_imagen(start_index=self.current_index)
    
    def undo_mitochondria(self):
        self.mascara_mitocondrias=self.mascara_mitocondrias_undo
        self.volumen_con_mitocondrias=self.volumen_con_mitocondrias_undo
        self.volumen_con_mitocondrias_sin_neuronas=self.volumen_con_mitocondrias_sin_neuronas_undo
        
        self.volumen_a_visualizar=self.volumen_con_mitocondrias
        self.visualizar_imagen(start_index=self.current_index)
        #mitocondrias_state = 2 if self.checkBox_mito.isChecked() else 0
        #self.estado_mascara_mitocondrias(mitocondrias_state)

    def on_mouse_clicked_vesicles(self, event):
        config_file = os.path.join(self.path_carpeta, f'{self.project_name}.txt')
        if os.path.exists(config_file):
            with open(config_file, 'r') as file:
                for line in file:
                    if line.startswith('#Cantidad de vesiculas:'):
                        self.cantidad_vesiculas = float(line.split(':')[1].strip())
        self.mascara_center_vesicles_undo=(self.mascara_center_vesicles).copy()
        self.mascara_area_vesicles_undo=(self.mascara_area_vesicles).copy()
        self.mascara_area_vesicles_etiquetada_undo=(self.mascara_area_vesicles_etiquetada).copy()

        self.volumen_con_areaycentrovesiculas_sinmito_undo=(self.volumen_con_areaycentrovesiculas_sinmito).copy()

        # Get the position of the mouse click in the image
        mouse_point = self.image_widget.getView().mapSceneToView(event.scenePos())
        self.x, self.y = int(mouse_point.x()), int(mouse_point.y())
        # Print the coordinates
        print(f"Mouse clicked at: x={self.x}, y={self.y}")

        #center -- un punto blanco en x,y
        self.mascara_center_vesicles[self.current_index,self.x,self.y]=255

        #area -- etiquetada y blanca
        #crear mascara con un circulo de radio 5 pixeles para que no se escape el reg growing
        self.mascara_limite=np.zeros((self.mascara_center_vesicles.shape[1],self.mascara_center_vesicles.shape[2]))
        #circulo radio 6
        radius=6
        x_center, y_center = self.y, self.x
        height, width =self.mascara_limite.shape

        # Limitar el rango de y para no salir de los límites de la imagen
        y_start = max(y_center - radius, 0)
        y_end = min(y_center + radius, height - 1)

        # Limitar el rango de x para no salir de los límites de la imagen
        x_start = max(x_center - radius, 0)
        x_end = min(x_center + radius, width - 1)

        # Solo recorrer el cuadrado delimitado por el radio alrededor del centro
        for y in range(y_start, y_end + 1):
            for x in range(x_start, x_end + 1):
                if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                    self.mascara_limite[y, x] = 255

        new_mask_etiquetada, self.cantidad_vesiculas=region_growing_etiquetado(self.tiff[self.current_index,:,:],self.mascara_area_vesicles_etiquetada[self.current_index,:,:],self.mascara_limite,[(self.x,self.y)],30,self.cantidad_vesiculas)
        new_mask_white=region_growing(self.tiff[self.current_index,:,:],self.mascara_area_vesicles[self.current_index,:,:],self.mascara_limite,[(self.x,self.y)],30)
        self.mascara_area_vesicles[self.current_index,:,:]=new_mask_white
        self.mascara_area_vesicles_etiquetada[self.current_index,:,:]=new_mask_etiquetada

        project_details_path = os.path.join(self.path_carpeta, f'{self.project_name}.txt')

        # Leer todas las líneas del archivo
        with open(project_details_path, 'r') as f:
            lines = f.readlines()

        # Reemplazar la línea que contiene "Cantidad de vesiculas"
        with open(project_details_path, 'w') as f:
            for line in lines:
                if line.startswith("Cantidad de vesiculas:"):
                    # Modificar la línea con el nuevo valor
                    f.write(f'#Cantidad de vesiculas: {self.cantidad_vesiculas}\n')
                else:
                    f.write(line)

        self.volumen_con_areavesiculas_sinmito=self.colorear_areavesiculas(self.volumen_con_mascaras,self.mascara_area_vesicles,RGB=False)
        self.volumen_con_areaycentrovesiculas_sinmito=self.colorear_centrovesiculas(self.volumen_con_areavesiculas_sinmito,self.mascara_center_vesicles,RGB=True)
        
        self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas_sinmito
        self.visualizar_imagen(start_index=self.current_index)
    
    def on_mouse_clicked_delete_vesicles(self, event):
        self.mascara_center_vesicles_undo=(self.mascara_center_vesicles).copy()
        self.mascara_area_vesicles_undo=(self.mascara_area_vesicles).copy()
        self.mascara_area_vesicles_etiquetada_undo=(self.mascara_area_vesicles_etiquetada).copy()

        self.volumen_con_areaycentrovesiculas_sinmito_undo=(self.volumen_con_areaycentrovesiculas_sinmito).copy()

        # Get the position of the mouse click in the image
        mouse_point = self.image_widget.getView().mapSceneToView(event.scenePos())
        self.x, self.y = int(mouse_point.x()), int(mouse_point.y())
        # Print the coordinates
        print(f"Mouse clicked at: x={self.x}, y={self.y}")

        #center -- un punto blanco en x,y
        #self.mascara_center_vesicles[self.current_index,self.x,self.y]=255

        #area -- etiquetada y blanca
        #crear mascara con un circulo de radio 5 pixeles para que no se escape el reg growing
        self.mascara_limite=np.zeros((self.mascara_center_vesicles.shape[1],self.mascara_center_vesicles.shape[2]))
        #circulo radio 6
        radius=6
        x_center, y_center = self.y, self.x
        height, width =self.mascara_limite.shape

        # Limitar el rango de y para no salir de los límites de la imagen
        y_start = max(y_center - radius, 0)
        y_end = min(y_center + radius, height - 1)

        # Limitar el rango de x para no salir de los límites de la imagen
        x_start = max(x_center - radius, 0)
        x_end = min(x_center + radius, width - 1)

        # Solo recorrer el cuadrado delimitado por el radio alrededor del centro
        for y in range(y_start, y_end + 1):
            for x in range(x_start, x_end + 1):
                if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                    self.mascara_limite[y, x] = 255
        print("intensidad vesicula:", self.mascara_area_vesicles_etiquetada[self.current_index,self.x,self.y])
        new_mask_etiquetada, new_mask_centers=region_growing_etiquetado_delete(self.tiff[self.current_index,:,:],self.mascara_area_vesicles_etiquetada[self.current_index,:,:],self.mascara_center_vesicles[self.current_index,:,:],self.mascara_limite,[(self.x,self.y)],self.mascara_area_vesicles_etiquetada[self.current_index,self.x,self.y])
        new_mask_white=region_growing_delete(self.tiff[self.current_index,:,:],self.mascara_area_vesicles[self.current_index,:,:],self.mascara_limite,[(self.x,self.y)])
        self.mascara_area_vesicles[self.current_index,:,:]=new_mask_white
        self.mascara_center_vesicles[self.current_index,:,:]=new_mask_centers
        self.mascara_area_vesicles_etiquetada[self.current_index,:,:]=new_mask_etiquetada

        self.volumen_con_areavesiculas_sinmito=self.colorear_areavesiculas(self.volumen_con_mascaras,self.mascara_area_vesicles,RGB=False)
        self.volumen_con_areaycentrovesiculas_sinmito=self.colorear_centrovesiculas(self.volumen_con_areavesiculas_sinmito,self.mascara_center_vesicles,RGB=True)
        
        self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas_sinmito
        self.visualizar_imagen(start_index=self.current_index)
    
    def undo_vesicles(self):
        self.mascara_center_vesicles=self.mascara_center_vesicles_undo
        self.mascara_area_vesicles=self.mascara_area_vesicles_undo
        self.mascara_area_vesicles_etiquetada=self.mascara_area_vesicles_etiquetada_undo

        self.volumen_con_areaycentrovesiculas_sinmito=self.volumen_con_areaycentrovesiculas_sinmito_undo
        project_details_path = os.path.join(self.path_carpeta, f'{self.project_name}.txt')

        # Leer todas las líneas del archivo
        with open(project_details_path, 'r') as f:
            lines = f.readlines()

        # Reemplazar la línea que contiene "Cantidad de vesiculas"
        with open(project_details_path, 'w') as f:
            for line in lines:
                if line.startswith("Cantidad de vesiculas:"):
                    # Modificar la línea con el nuevo valor
                    f.write(f'#Cantidad de vesiculas: {self.cantidad_vesiculas-1}\n')
                else:
                    f.write(line)
        
        self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas_sinmito
        self.visualizar_imagen(start_index=self.current_index)
        #area_ves_state = 2 if self.checkBox_area_ves.isChecked() else 0
        #self.estado_mascara_area_vesicle(area_ves_state)
    
    def calculate_metrics(self):
        volume_mito, surface_mito, mci, mbi = metrics_mitocondria(self.mascara_etiquetada_mitochondria, self.pixel_size, self.z_scale)
        output_path = os.path.join(self.path_carpeta, f'{self.project_name}.xlsx')
        
        # Filtrar los datos para incluir solo mitocondrias con volumen y superficie > 0
        filtered_data = {
            'Mitochondria': [],
            'Volume [nm3]': [],
            'Surface Area [nm2]': [],
            'MCI': [],
            'MBI': []
        }
        
        # Iterar desde el índice 1 (saltando el fondo)
        for i in range(1, len(volume_mito)):
            if volume_mito[i] > 0 and surface_mito[i] > 0:
                filtered_data['Mitochondria'].append(f'Mitochondria {i}')
                filtered_data['Volume [nm3]'].append(volume_mito[i])
                filtered_data['Surface Area [nm2]'].append(surface_mito[i])
                filtered_data['MCI'].append(mci[i])
                filtered_data['MBI'].append(mbi[i])
        
        # Crear DataFrame con los datos filtrados
        df = pd.DataFrame(filtered_data)
        df.to_excel(output_path, index=False)
        print('Se ha guardado el excel con las métricas de mitocondrias con volumen y superficie > 0')
    
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())

