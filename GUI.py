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
        loadUi(r'.\prueba_interfaz_vesicles.ui', self) #path to .ui
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
        self.actionVesicles_Label.triggered.connect(self.label_ves)
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
        self.tiff=tifffile.imread(path_volumen)
        print("dim tiff: ",self.tiff.shape)
        if self.tiff.ndim != 3:
            with tifffile.TiffFile(path_volumen) as tif:
                pages = [page.asarray() for page in tif.pages]
                self.tiff = np.stack(pages, axis=0)
        self.tiff=np.transpose(self.tiff, axes=(0,2,1))
        print("dim tiff: ",self.tiff.shape)
        
        self.volumen_a_visualizar=self.tiff
        self.image_widget.getView().setAspectLocked(True)
        self.image_widget.getView().autoRange(padding=0.0)
        self.image_widget.timeLine.setPen((255,255,0,200))
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
        if self.mascara.shape != self.tiff.shape:
            QMessageBox.warning(self, "Error", "Mask and volume have different dimensions.")
            return
        self.mascara_exists=True
        oscurecido_volumen = (self.tiff * 0.5).astype(np.uint8)
        self.volumen_con_mascaras = np.where(self.mascara >= 250, self.tiff, oscurecido_volumen).astype(np.uint8)
        
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
        
        if self.checkBox_area_ves.isChecked() and self.checkBox_center_ves.isChecked():
            self.checkBox_neurons.setChecked(True)
            if self.checkBox_mito.isChecked():
                self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas
            else:
                self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas_sinmito
        self.visualizar_imagen(start_index=self.current_index)
        
            
    def cargar_mascara_mitocondrias(self, path_mascara):
        self.mascara_mitocondrias = (tifffile.imread(path_mascara)).astype(np.uint8)
        self.mascara_mitocondrias = np.transpose(self.mascara_mitocondrias, axes=(0, 2, 1))
        if self.mascara_mitocondrias.shape != self.tiff.shape:
            QMessageBox.warning(self, "Error", "Mitochondria mask and volume have different dimensions")
            return
        self.mascara_mitocondrias_exists=True
    
    def colorear_mascara_mitocondrias(self, volumen, mask, RGB, un_slide=False):  
        alpha = 0.2
        red_color = np.array([253, 60, 20], dtype=np.uint8)
        if not RGB:
            volumen = np.stack([volumen] * 3, axis=-1)
        volumen_con_mitocondrias = np.copy(volumen)
        red_overlay = np.zeros_like(volumen)
        if un_slide:
            red_overlay[(mask >= 250)  & (self.tiff[self.current_index,:,:] <=205)] = red_color
        else:
            red_overlay[(mask >= 250)  & (self.tiff <=205)] = red_color
        volumen_con_mitocondrias = (alpha * red_overlay + volumen).astype(np.uint8)
        return volumen_con_mitocondrias
    
    def colorear_areavesiculas(self, volumen, mask, RGB):  
        alpha = 0.2
        blue_color = np.array([20, 60, 253], dtype=np.uint8)  # Color azul

        if not RGB:
            volumen = np.stack([volumen] * 3, axis=-1)

        volumen_con_areavesiculas = np.copy(volumen)
        blue_overlay = np.zeros_like(volumen)
        blue_overlay[(mask >= 250) & (self.tiff <=205)] = blue_color
        volumen_con_areavesiculas = (alpha * blue_overlay + volumen).astype(np.uint8)
        return volumen_con_areavesiculas
    
    def colorear_centrovesiculas(self, volumen, mask, RGB):  
        alpha = 0.9
        green_color = np.array([60, 255, 0], dtype=np.uint8)  # Color amarillo

        if not RGB:
            volumen = np.stack([volumen] * 3, axis=-1)
        volumen_con_centrovesiculas = np.copy(volumen)

        yellow_overlay = np.copy(volumen)
        yellow_overlay[mask >= 250] = green_color
        
        volumen_con_centrovesiculas = (yellow_overlay).astype(np.uint8)
        return volumen_con_centrovesiculas


    def visualizar_imagen(self, start_index):
        self.image_widget.getView().setAspectLocked(True)
        self.image_widget.getView().autoRange(padding=0.0)
        self.image_widget.timeLine.setPen((255,255,0,200))
        self.image_widget.setImage(self.volumen_a_visualizar)
        self.image_widget.setCurrentIndex(start_index)
        
    def on_time_line_changed(self):
        self.current_index = int(self.image_widget.currentIndex)
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
        self.project_name, ok = QInputDialog.getText(self, 'New Project', 'Project name:')
        if not ok or not self.project_name:
            return 
        pixel_size, ok = QInputDialog.getDouble(self, 'Píxel size', 'Insert pixel size (nm):', decimals=2)
        if not ok:
            return  
        self.pixel_size = pixel_size
        
        z_scale, ok = QInputDialog.getDouble(self, 'Z scale', 'Insert z-scale:', decimals=2)
        if not ok:
            return
        self.z_scale = z_scale
        
        project_dir = QFileDialog.getExistingDirectory(self, 'Select folder where project will be created')
        if not project_dir:
            return
        project_dir = os.path.join(project_dir, self.project_name)
        
        try:
            os.makedirs(project_dir)
        except FileExistsError:
            QMessageBox.warning(self, "Error", "The project already exists.")
            return
        
        volume_file, _ = QFileDialog.getOpenFileName(self, "Select volume to analize", "", "Archivos (*.tif *.tiff)")
        if not volume_file:
            QMessageBox.warning(self, "Error", "A volume must be selected.")
            return
        

        mask_file, _ = QFileDialog.getOpenFileName(self, "In case it exists, select the neuron mask", "", "Archivos (*.tif *.tiff)")
        if not mask_file:
            QMessageBox.warning(self, "Warning!", "No neuron mask has been chosen")
            #pass
        else:
            mask_dest = os.path.join(project_dir, 'mascara_neuronas' + os.path.splitext(mask_file)[1])
            shutil.copy2(mask_file, mask_dest)
        volume_dest = os.path.join(project_dir, 'volumen' + os.path.splitext(volume_file)[1])
        
        shutil.copy2(volume_file, volume_dest)
        project_details_path = os.path.join(project_dir, f'{self.project_name}.txt')
        with open(project_details_path, 'w') as f:
            f.write(f'#Project Name: {self.project_name}\n')
            f.write(f'# Pixel Size (nm): {self.pixel_size}\n')
            f.write(f'# Z Scale: {self.z_scale}\n')
        
        QMessageBox.information(self, "Success", f"The project '{self.project_name}' has been created and the files have been copied.")
        self.analizar_carpeta(project_dir)
        self.path_carpeta=project_dir
        parent_dir = os.path.dirname(project_dir)
        self.parent_path=parent_dir
        self.setWindowTitle(self.project_name)

    def open_project(self):
        project_dir = QFileDialog.getExistingDirectory(self, 'Open folder of existing project')
        if not project_dir:
            return None
        self.project_name = os.path.basename(project_dir)
        volumen_path = None

        for archivo in os.listdir(project_dir):
            archivo_path = os.path.join(project_dir, archivo)
            
            if archivo.lower() == 'volumen.tif' or archivo.lower() == 'volumen.tiff':
                volumen_path = archivo_path

        if not volumen_path:
            QMessageBox.warning(self, "Error", "No files 'volumen.tif' or 'volumen.tiff' have been found in the selected folder.")
            return None
        
        config_file = os.path.join(project_dir, f'{self.project_name}.txt')
        if os.path.exists(config_file):
            with open(config_file, 'r') as file:
                for line in file:
                    if line.startswith('# Pixel Size (nm):'):
                        self.pixel_size = float(line.split(':')[1].strip())
                    elif line.startswith('# Z Scale:'):
                        self.z_scale = float(line.split(':')[1].strip())
        else:
            QMessageBox.warning(self, "Warning!", "Configuration file has not been found in the project.")
        
        parent_dir = os.path.dirname(project_dir)
        self.analizar_carpeta(project_dir)
        self.path_carpeta=project_dir 
        self.parent_path=parent_dir
        self.setWindowTitle(self.project_name)

    def analizar_carpeta(self, project_dir):
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
            if archivo.lower() == 'mascara_mitocondrias_etiquetada.tif':
                path_mascara_mitocondrias_etiquetada = archivo_path
                self.mascara_etiquetada_mitochondria_exists=True
                self.mascara_etiquetada_mitochondria = tifffile.imread(path_mascara_mitocondrias_etiquetada)
                self.mascara_etiquetada_mitochondria = np.transpose(self.mascara_etiquetada_mitochondria, axes=(0, 2, 1))

        for archivo in os.listdir(project_dir):
            archivo_path = os.path.join(project_dir, archivo)
            if archivo.lower() == 'mascara_area_vesicles.tif':
                path_area_vesicles = archivo_path
                self.mascara_area_vesicles_exists=True
                self.checkBox_area_ves.setChecked(True)
                self.mascara_area_vesicles=tifffile.imread(path_area_vesicles)
                self.mascara_area_vesicles = np.transpose(self.mascara_area_vesicles, axes=(0, 2, 1))
        
        for archivo in os.listdir(project_dir):
            archivo_path = os.path.join(project_dir, archivo)
            if archivo.lower() == 'mascara_center_vesicles.tif':
                path_center_vesicle = archivo_path
                self.mascara_center_vesicle_exists=True
                self.checkBox_center_ves.setChecked(True)
                self.mascara_center_vesicles=tifffile.imread(path_center_vesicle)
                self.mascara_center_vesicles = np.transpose(self.mascara_center_vesicles, axes=(0, 2, 1))
                self.volumen_con_centrovesiculas=self.colorear_centrovesiculas(self.volumen_con_mitocondrias,self.mascara_center_vesicles,RGB=True)
                self.volumen_con_areavesiculas=self.colorear_areavesiculas(self.volumen_con_mitocondrias,self.mascara_area_vesicles,RGB=True)
                self.volumen_con_areaycentrovesiculas=self.colorear_centrovesiculas(self.volumen_con_areavesiculas,self.mascara_center_vesicles,RGB=True)
                self.volumen_con_areavesiculas_sinmito=self.colorear_areavesiculas(self.volumen_con_mascaras,self.mascara_area_vesicles,RGB=False)
                self.volumen_con_centrovesiculas_sinmito=self.colorear_centrovesiculas(self.volumen_con_mascaras, self.mascara_center_vesicles, RGB=False )
                self.volumen_con_areaycentrovesiculas_sinmito=self.colorear_centrovesiculas(self.volumen_con_areavesiculas_sinmito,self.mascara_center_vesicles,RGB=True)
                self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas
        
        for archivo in os.listdir(project_dir):
            archivo_path = os.path.join(project_dir, archivo)
            if archivo.lower() == 'mascara_area_vesicles_etiquetada.tif':
                path_area_vesicles_etiquetada = archivo_path
                self.mascara_area_vesicles_etiquetada_exists=True
                self.mascara_area_vesicles_etiquetada=tifffile.imread(path_area_vesicles_etiquetada)
                self.mascara_area_vesicles_etiquetada = np.transpose(self.mascara_area_vesicles_etiquetada, axes=(0, 2, 1))
        
        for archivo in os.listdir(project_dir):
            archivo_path = os.path.join(project_dir, archivo)
            if archivo.lower() == 'mascara_center_vesicles_etiquetada.tif':
                path_center_vesicles_etiquetada = archivo_path
                self.mascara_center_vesicles_etiquetada_exists=True
                self.mascara_center_vesicles_etiquetada=tifffile.imread(path_center_vesicles_etiquetada)
                self.mascara_center_vesicles_etiquetada = np.transpose(self.mascara_center_vesicles_etiquetada, axes=(0, 2, 1))
                
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
            QMessageBox.warning(self, "Error", "There must be a neuron mask to predict mitochondria and vesicles.")
            return None
        if not self.multiplied_exists or (self.multiplied.shape != self.tiff.shape):
            self.multiplied=multiply_mask(self.tiff, self.mascara, self.path_carpeta, True)
    
    def predict_mitochondria (self):
        self.cargar_multiplied()
        self.mascara_mitocondrias=predict_mito(self.multiplied, self.parent_path, self.path_carpeta) 
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
            dialog = QDialog(self)
            dialog.setWindowTitle("Seleccionar método de predicción")
            dialog.setGeometry(150, 150, 300, 150)

            layout = QVBoxLayout(dialog)
            option1 = QRadioButton("Opción 1 - Prediccion de vesiculas utilizando bin1 - Accuracy: xx%", dialog)
            option2 = QRadioButton("Opción 2 - Prediccion de vesiculas utilizando bin5 - Accuracy: xx%", dialog)
            option1.setChecked(True)
            layout.addWidget(option1)
            layout.addWidget(option2)
            button = QPushButton("Confirmar", dialog)
            button.clicked.connect(dialog.accept)  # Cierra el diálogo con QDialog.Accepted
            layout.addWidget(button)
            result = dialog.exec_()
            if result == QDialog.Accepted:
                if option1.isChecked():
                    QMessageBox.information(self, "Selección", "Option 1 selected: Prediction with bin1")
                    self.predict_ves_bin1()
                else:
                    QMessageBox.information(self, "Selección", "Option 2 selected: Prediction with bin5")
                    self.predict_ves_bin5()
        else:
            QMessageBox.warning(self, "Error", "Vesicles cannot be predicted without a neuron mask. First, predict neurons")
            return
    
    def predict_ves_bin1(self):
        pixel_size_bin1, ok = QInputDialog.getDouble(self, 'Pixel size bin 1', 'Insert pixel size (nm):', decimals=2)
        if not ok:
            return  
        
        ruta_bin1, ok = QInputDialog.getText(self, 'Path to bin 1', 'Insert complete path to bin1 file, including its name.')
        if not ok:
            return

        project_details_bin1_path = os.path.join(self.path_carpeta, f'{self.project_name}_bin1.txt')
        with open(project_details_bin1_path, 'w') as f:
            f.write(f'# Project Name: {self.project_name}_bin1\n')
            f.write(f'# Pixel Size bin 1 (nm): {pixel_size_bin1}\n')
            f.write(f'# Ruta de acceso + nombre: {ruta_bin1}')
            
        config_file_bin1 = os.path.join(self.path_carpeta, f'{self.project_name}_bin1.txt')
        if os.path.exists(config_file_bin1):
            with open(config_file_bin1, 'r') as file:
                for line in file:
                    if line.startswith('# Pixel Size bin 1 (nm):'):
                        self.pixel_size = float(line.split(':')[1].strip())
                    elif line.startswith('# Ruta de acceso + nombre:'):
                        inicio = line.find("r'") + 2
                        self.ruta_bin1 = line[inicio:].strip().strip("'")
        self.cargar_multiplied()
        self.bin1=tifffile.imread(self.ruta_bin1)
        self.bin1=np.transpose(self.bin1, axes=(0,2,1))
        
        tamanobin1=self.bin1[0].shape
        tamanobin5=self.mascara[0].shape
        
        multiplied=False
        if multiplied==False:
            self.mascara_bin1 = np.zeros((self.bin1.shape[0], self.bin1.shape[1], self.bin1.shape[2]), dtype=np.uint8)

            for i in range(self.bin1.shape[0]):
                self.mascara_bin1[i] = cv2.resize(self.mascara[i],((tamanobin1[1],tamanobin1[0])), interpolation=cv2.INTER_NEAREST_EXACT)
            
            self.multiplied_bin1=multiply_mask(self.bin1, self.mascara_bin1, self.path_carpeta, False) #se crea pero no se guarda
        else:
            self.multiplied_bin1=self.bin1  
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
            QMessageBox.warning(self, "Error", "Vesicles cannot be labeled without a prediction, first predict vesicles")
            return

    def predict_neurons(self):
        self.mascara,self.mascara_mitocondrias=predict_neurons(self.tiff, self.parent_path, self.path_carpeta)
        self.checkBox_mito.setChecked(True)
        self.checkBox_neurons.setChecked(True)
        self.mascara_exists=True
        self.mascara_mitocondrias_exists=True
        oscurecido_volumen = (self.tiff * 0.5).astype(np.uint8)
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
        if self.mouse_click_connection is not None:
            if self.label_tool.text()=='You are adding mitochondria':
                self.image_widget.scene.sigMouseClicked.disconnect(self.on_mouse_clicked)
            if self.label_tool.text()=='You are adding vesicle':
                self.image_widget.scene.sigMouseClicked.disconnect(self.on_mouse_clicked_vesicles)
            self.mouse_click_connection = None
        self.label_tool.setText('')
    
    def finish_delete(self):
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
        mouse_point = self.image_widget.getView().mapSceneToView(event.scenePos())
        self.x, self.y = int(mouse_point.x()), int(mouse_point.y())
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
        mouse_point = self.image_widget.getView().mapSceneToView(event.scenePos())
        self.x, self.y = int(mouse_point.x()), int(mouse_point.y())
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
        mouse_point = self.image_widget.getView().mapSceneToView(event.scenePos())
        self.x, self.y = int(mouse_point.x()), int(mouse_point.y())
        self.mascara_center_vesicles[self.current_index,self.x,self.y]=255
        self.mascara_limite=np.zeros((self.mascara_center_vesicles.shape[1],self.mascara_center_vesicles.shape[2]))
        radius=6
        x_center, y_center = self.y, self.x
        height, width =self.mascara_limite.shape
        y_start = max(y_center - radius, 0)
        y_end = min(y_center + radius, height - 1)
        x_start = max(x_center - radius, 0)
        x_end = min(x_center + radius, width - 1)
        for y in range(y_start, y_end + 1):
            for x in range(x_start, x_end + 1):
                if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                    self.mascara_limite[y, x] = 255

        new_mask_etiquetada, self.cantidad_vesiculas=region_growing_etiquetado(self.tiff[self.current_index,:,:],self.mascara_area_vesicles_etiquetada[self.current_index,:,:],self.mascara_limite,[(self.x,self.y)],30,self.cantidad_vesiculas)
        new_mask_white=region_growing(self.tiff[self.current_index,:,:],self.mascara_area_vesicles[self.current_index,:,:],self.mascara_limite,[(self.x,self.y)],30)
        self.mascara_area_vesicles[self.current_index,:,:]=new_mask_white
        self.mascara_area_vesicles_etiquetada[self.current_index,:,:]=new_mask_etiquetada

        project_details_path = os.path.join(self.path_carpeta, f'{self.project_name}.txt')
        with open(project_details_path, 'r') as f:
            lines = f.readlines()
        with open(project_details_path, 'w') as f:
            for line in lines:
                if line.startswith("Cantidad de vesiculas:"):
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

        mouse_point = self.image_widget.getView().mapSceneToView(event.scenePos())
        self.x, self.y = int(mouse_point.x()), int(mouse_point.y())
        self.mascara_limite=np.zeros((self.mascara_center_vesicles.shape[1],self.mascara_center_vesicles.shape[2]))
        radius=6
        x_center, y_center = self.y, self.x
        height, width =self.mascara_limite.shape
        y_start = max(y_center - radius, 0)
        y_end = min(y_center + radius, height - 1)
        x_start = max(x_center - radius, 0)
        x_end = min(x_center + radius, width - 1)
        for y in range(y_start, y_end + 1):
            for x in range(x_start, x_end + 1):
                if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                    self.mascara_limite[y, x] = 255
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

        with open(project_details_path, 'r') as f:
            lines = f.readlines()

        with open(project_details_path, 'w') as f:
            for line in lines:
                if line.startswith("Cantidad de vesiculas:"):
                    f.write(f'#Cantidad de vesiculas: {self.cantidad_vesiculas-1}\n')
                else:
                    f.write(line)
        
        self.volumen_a_visualizar=self.volumen_con_areaycentrovesiculas_sinmito
        self.visualizar_imagen(start_index=self.current_index)
    
    def calculate_metrics(self):
        volume_mito, surface_mito, mci, mbi = metrics_mitocondria(self.mascara_etiquetada_mitochondria, self.pixel_size, self.z_scale)
        output_path = os.path.join(self.path_carpeta, f'{self.project_name}.xlsx')
        filtered_data = {
            'Mitochondria': [],
            'Volume [nm3]': [],
            'Surface Area [nm2]': [],
            'MCI': [],
            'MBI': []
        }
        
        for i in range(1, len(volume_mito)):
            if volume_mito[i] > 0 and surface_mito[i] > 0:
                filtered_data['Mitochondria'].append(f'Mitochondria {i}')
                filtered_data['Volume [nm3]'].append(volume_mito[i])
                filtered_data['Surface Area [nm2]'].append(surface_mito[i])
                filtered_data['MCI'].append(mci[i])
                filtered_data['MBI'].append(mbi[i])
        
        df = pd.DataFrame(filtered_data)
        df.to_excel(output_path, index=False)    
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())

