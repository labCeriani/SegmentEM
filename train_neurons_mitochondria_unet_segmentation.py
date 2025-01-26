"""
Multiclass semantic segmentation using U-net, Attention Unet and Att Res Unet
Images and masks are divided into patches of 256x256. 
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from datetime import datetime 
import cv2
from PIL import Image
from keras import backend, optimizers
import glob
from keras.utils import normalize

#Resizing images, if needed
SIZE_X = 256
SIZE_Y = 256
n_classes=3 #Number of classes for segmentation, if uniclass change to 1

#Capture training image info as a list
train_images = []
pathimgzt2=r'\Users\ulabceriani\Documents\ITBA\ITBA_ZT2\patches_multiplied_img_zt2'
for directory_path in glob.glob(pathimgzt2):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        img = cv2.imread(img_path, 0)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)
pathimgzt14=r'\Users\ulabceriani\Documents\ITBA\ITBA_ZT14\patches_multiplied_img_zt14'
for directory_path in glob.glob(pathimgzt14):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        img = cv2.imread(img_path, 0)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)

#Convert list to array for machine learning processing        
train_images = np.array(train_images)

#Capture mask/label info as a list

#################################################
#if uniclass:
'''
train_masks = [] 
pathmaskszt2=r'\Users\ulabceriani\Documents\ITBA\ITBA_ZT2\patches_multiplied_mask_zt2'
for directory_path in glob.glob(pathmaskszt2):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = cv2.imread(mask_path, 0)
        # Crear una máscara binaria donde los píxeles son 0 o 255
        mask_aux = np.where(mask >= 100, 255, 0).astype(np.uint8)
        # Redimensionar la máscara a las dimensiones deseadas
        mask = cv2.resize(mask_aux, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
        # Agregar la máscara procesada a la lista
        train_masks.append(mask)
pathmaskszt14=r'\Users\ulabceriani\Documents\ITBA\ITBA_ZT14\patches_multiplied_mask_zt14'
for directory_path in glob.glob(pathmaskszt14):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = cv2.imread(mask_path, 0)
        # Crear una máscara binaria donde los píxeles son 0 o 255
        mask_aux = np.where(mask >= 100, 255, 0).astype(np.uint8)
        # Redimensionar la máscara a las dimensiones deseadas
        mask = cv2.resize(mask_aux, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
        # Agregar la máscara procesada a la lista
        train_masks.append(mask)
'''
#################################################
#if multiclass:
#Capture mask/label info as a list
train_masks = [] 
for directory_path in glob.glob("/Users/catalinainsussarry/Downloads/combined_patches_mask_zt2"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = cv2.imread(mask_path, 0)
        # Create a mask where pixels are either 255 or 120
        mask_aux = np.where((mask == 255) | (mask == 120), 255, 0).astype(np.uint8)
        # Apply the mask to the original image
        result_image = cv2.bitwise_and(mask, mask, mask=mask_aux) 
        mask=result_image    
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        train_masks.append(mask)

for directory_path in glob.glob("/Users/catalinainsussarry/Downloads/combined_patches_mask_zt14"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = cv2.imread(mask_path, 0)
        # Create a mask where pixels are either 255 or 120
        mask_aux = np.where((mask == 255) | (mask == 120), 255, 0).astype(np.uint8)
        # Apply the mask to the original image
        result_image = cv2.bitwise_and(mask, mask, mask=mask_aux)
        mask=result_image        
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        train_masks.append(mask)

#################################################
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)
#train_masks.sort()
print("multiclass train",len(train_images))
print("multiclass train",len(train_masks))

print(np.unique(train_masks))

###############################################
#If multiclass:
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

print(np.unique(train_masks_encoded_original_shape))


train_images = np.expand_dims(train_images, axis=3)
train_images = normalize(train_images, axis=1)

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

#################################################

#if uniclass:
'''
#Normalize images
train_images = np.expand_dims(normalize(np.array(train_images), axis=1),3)
#train_images = np.array(train_images)/255.
#train_images = np.expand_dims(train_images, axis=3)
#train_images = normalize(train_images, axis=1)
#D not normalize masks, just rescale to 0 to 1.
train_masks_input = np.expand_dims((np.array(train_masks)),3) /255.

'''

#################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size = 0.2, random_state = 0)
#X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0)

#Further split training data t a smaller subset for quick testing of models
#X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.1, random_state = 0)

print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 

#################################################
#if multiclass:
from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))
print("Forma de X_train:", X_train.shape)
print("Forma de y_train_cat:", y_train_cat.shape)
#################################################

#Sanity check, view few mages
import random
import numpy as np

image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(X_train[image_number], cmap='gray')
plt.subplot(122)
plt.imshow(y_train[image_number], cmap='gray')
plt.show()

#######################################
#Parameters for model

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
num_labels = 3 #if uniclass change to 1
input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
print("input shape", input_shape)
batch_size = 8

#FOCAL LOSS AND DICE METRIC
#Focal loss helps focus more on tough to segment classes.
from focal_loss import BinaryFocalLoss

###############################################################################


#Try various models: Unet, Attention_UNet, and Attention_ResUnet
#Rename original python file from 224_225_226_models.py to models.py
from models import Attention_ResUNet, UNet, Attention_UNet, dice_coef, dice_coef_loss, jacard_coef



#UNet

unet_model = UNet(input_shape, num_labels)
unet_model.compile(optimizer='adam', loss=BinaryFocalLoss(gamma=2), metrics=['accuracy', jacard_coef])


print(unet_model.summary())

start1 = datetime.now() 
unet_history = unet_model.fit(X_train, y_train_cat, verbose=1,batch_size = batch_size,validation_data=(X_test, y_test_cat), shuffle=False, epochs=50)

stop1 = datetime.now()
#Execution time of the model 
execution_time_Unet = stop1-start1
print("UNet execution time is: ", execution_time_Unet)

unet_model.save(r'/Users/ulabceriani/Documents/ITBA/multiclass_UNet_50epochs_B_focal_2.hdf5')
#____________________________________________

#Attention UNet

att_unet_model = Attention_UNet(input_shape, num_labels)

att_unet_model.compile(optimizer='adam', loss=BinaryFocalLoss(gamma=2), metrics=['accuracy', jacard_coef])


print(att_unet_model.summary())
start2 = datetime.now() 
att_unet_history = att_unet_model.fit(X_train, y_train, verbose=1, batch_size = batch_size, validation_data=(X_test, y_test), shuffle=False,epochs=50)
stop2 = datetime.now()
#Execution time of the model 
execution_time_Att_Unet = stop2-start2
print("Attention UNet execution time is: ", execution_time_Att_Unet)

att_unet_model.save(r'\Users\ulabceriani\Documents\ITBA\multiclass_Attention_UNet_50epochs_B_focal_2.hdf5')


#Attention Residual Unet

att_res_unet_model = Attention_ResUNet(input_shape)

att_res_unet_model.compile(optimizer='adam', loss=BinaryFocalLoss(gamma=2), 
              metrics=['accuracy', jacard_coef])


# att_res_unet_model.compile(optimizer=Adam(lr = 1e-3), loss='binary_crossentropy', 
#               metrics=['accuracy', jacard_coef])

print(att_res_unet_model.summary())


start3 = datetime.now() 
att_res_unet_history = att_res_unet_model.fit(X_train, y_train, #aca cambiar a y_train_cat y y_test_cat si es multiclass 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=50)
stop3 = datetime.now()

#Execution time of the model 
execution_time_AttResUnet = stop3-start3
print("Attention ResUnet execution time is: ", execution_time_AttResUnet)

att_res_unet_model.save(r'\Users\ulabceriani\Documents\ITBA\simple_class_AttResUnet_50epochs_B_focal_2.hdf5')



