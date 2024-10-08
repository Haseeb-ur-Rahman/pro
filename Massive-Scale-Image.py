# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 20:33:19 2022

@author: okokp
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.io import imread
import tensorflow

df = pd.read_csv('DataSet/Brain Tumor.csv')
print("",df)
plt.pie(
    list(df['Class'].value_counts()),
    labels=['No tumour','Has tumour']
)
plt.title("The percentage of the data we have in the dataset")
plt.show()

IMAGES_DATA_SET_DIR = 'DataSet/Images/'
IMAGES_PATHS = [f'{IMAGES_DATA_SET_DIR}Image{i}.jpg' for i in range(1,len(os.listdir(IMAGES_DATA_SET_DIR))+1)]
for ind,i in enumerate(IMAGES_PATHS[:9]):
    img = imread(i)
    plt.imshow(img)
    plt.subplot(3,3,ind+1)
    
df['Image Paths'] = df['Image'].map(lambda x:f'{IMAGES_DATA_SET_DIR}{x}.jpg')
print("",df)

from PIL import Image
df['Image Pixels'] = df['Image Paths'].map(
    lambda x:np.asarray( Image.open(x).resize((224,224)) )
)
print("",df)

from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import mobilenet_v2
image_list = [
    mobilenet_v2.preprocess_input(
        img_to_array(
            df['Image Pixels'][i].astype(np.float32)
            # Converting pixel integral data to suitable float data
        )
    ) for i in range(0,len(df))
]
X = np.array(image_list)
print(X.shape)

y = np.array(df['Class'])
print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import Sequential, layers

model = Sequential()
model.add(MobileNetV2(
    input_shape=(224, 224, 3),
    weights="imagenet",
    include_top=False
))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(
    1, # Number of outputs from each node of dense layer
    activation='sigmoid'
))
model.layers[0].trainable= False
# show model summary
model.summary()

model.compile(
    loss=tensorflow.keras.losses.binary_crossentropy,
    optimizer=tensorflow.keras.optimizers.SGD(learning_rate = 0.01),
    metrics=['accuracy']
)
model.fit(
    X_train,
    y_train,
    epochs=10,
    verbose=1,
    validation_data=(X_test,y_test)
)
model.save("model.h5")
print("model saved")


from tensorflow.keras.models import load_model
pretrained_cnn = load_model('model.h5')
eval_score = pretrained_cnn.evaluate(X_test,y_test)


print('Eval loss:{}\nEval accuracy:{}'.format(*eval_score))


y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int)


import yaml # To print dictionary decent pattern
from sklearn.metrics import confusion_matrix , classification_report
target_classes = ['No Tumor','Tumor']
print(yaml.dump(
        classification_report(y_test , y_pred , output_dict = True
                      , target_names=target_classes))
)