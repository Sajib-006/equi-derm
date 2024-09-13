import numpy as np
import pandas as pd
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import cv2
import os
from skimage import exposure
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.math import confusion_matrix
from sklearn.metrics import accuracy_score
from seaborn import heatmap
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from ast import literal_eval
from matplotlib.patches import Rectangle

img_size = 224
batch_size = 16

def load_process(img, img_size):
    img = load_img(img, target_size = (img_size, img_size))
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_image(img)
    return img

chex_weights_path = 'brucechou1983_CheXNet_Keras_0.3.0_weights.keras'

pre_model = DenseNet121(weights=None,
                                include_top=False,
                                input_shape=(img_size,img_size,3)
                               )
out = Dense(14, activation='sigmoid')(pre_model.output)
pre_model = Model(inputs=pre_model.input, outputs=out) 
pre_model.load_weights(chex_weights_path)
pre_model.trainable = False
x = pre_model.layers[-2].output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.1)(x)
output = Dense(2, activation='softmax')(x)
model = Model(pre_model.input, output)


model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

# model.compile(Adam(lr=1e-3),loss='binary_crossentropy',metrics='accuracy')


# training
rlr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.2, patience = 2, verbose = 1, 
                                min_delta = 1e-4, min_lr = 1e-4, mode = 'max')
es = EarlyStopping(monitor = 'val_accuracy', min_delta = 1e-4, patience = 5, mode = 'max', 
                    restore_best_weights = True, verbose = 1)

ckp = ModelCheckpoint('model.h5',monitor = 'val_accuracy',
                      verbose = 0, save_best_only = True, mode = 'max')

history = model.fit(
      train_generator,
      epochs=20,
      validation_data=valid_generator,
      callbacks=[es,rlr, ckp],
      verbose=1)

## finetuning
pre_model.trainable = True
model.compile(Adam(lr=1e-5),loss='binary_crossentropy',metrics='accuracy')
rlr2 = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.1, patience = 2, verbose = 1, 
                                min_delta = 1e-4, min_lr = 1e-7, mode = 'max')
es2 = EarlyStopping(monitor = 'val_accuracy', min_delta = 1e-4, patience = 7, mode = 'max', 
                    restore_best_weights = True, verbose = 1)
history2 = model.fit(
      train_generator,
      epochs=40,
      validation_data=valid_generator,
      callbacks=[es2,rlr2, ckp],
      verbose=1)

K.clear_session()

#performace
actual =  valid_generator.labels
preds = np.argmax(model.predict(valid_generator), axis=1)
cfmx = confusion_matrix(actual, preds)
acc = accuracy_score(actual, preds)


print ('Test Accuracy:', acc )
heatmap(cfmx, annot=True, cmap='plasma',
        xticklabels=['Normal','Opacity'],fmt='.0f', yticklabels=['Normal', 'Opacity'])
plt.show()

#hist = pd.DataFrame(history.history)
fig, (ax1, ax2) = plt.subplots(figsize=(12,12),nrows=2, ncols=1)
hist['loss'].plot(ax=ax1,c='k',label='training loss')
hist['val_loss'].plot(ax=ax1,c='r',linestyle='--', label='validation loss')
ax1.legend()
hist['accuracy'].plot(ax=ax2,c='k',label='training accuracy')
hist['val_accuracy'].plot(ax=ax2,c='r',linestyle='--',label='validation accuracy')
ax2.legend()
plt.show()