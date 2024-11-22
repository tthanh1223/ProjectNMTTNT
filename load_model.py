import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import pytesseract as pt
import plotly.express as px
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet



from glob import glob
from skimage import io
from shutil import copy
from keras.src.models import Model
from keras.src.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from keras.src.applications.inception_resnet_v2 import InceptionResNetV2
from keras.src.layers import Dense, Dropout, Flatten, Input
from keras.src.utils import load_img, img_to_array

model = tf.keras.models.load_model('./my_model.keras')
print('Model loaded Sucessfully')

path = 'DataSet/N49.jpeg'
def object_detection(path):

    # Read image
    image = load_img(path) # PIL object
    image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    image1 = load_img(path,target_size=(224,224))

    # Data preprocessing
    image_arr_224 = img_to_array(image1)/255.0 # Convert to array & normalized
    h,w,d = image.shape
    test_arr = image_arr_224.reshape(1,224,224,3)

    # Make predictions
    coords = model.predict(test_arr)

    # Denormalize the values
    denorm = np.array([w,w,h,h])
    coords = coords * denorm
    coords = coords.astype(np.int32)

    # Draw bounding on top the image
    xmin, xmax, ymin, ymax = coords[0]
    pt1 =(xmin,ymin)
    pt2 =(xmax,ymax)
    print(pt1, pt2)
    cv2.rectangle(image,pt1,pt2,(0,255,0),3)
    return image, coords

image, cods = object_detection(path)

fig = px.imshow(image)
fig.update_layout(width=700, height=500, margin=dict(l=10, r=10, b=10, t=10))
fig.show()

img = np.array(load_img(path))
xmin, xmax, ymin, ymax = cods[0]
roi = img[ymin:ymax,xmin:xmax]
fig = px.imshow(roi)
fig.update_layout(width=350, height=250, margin=dict(l=10, r=10, b=10, t=10))
fig.show()

pt.pytesseract.tesseract_cmd = r'.\Tesseract-OCR\tesseract.exe'
# extract text from image
text = pt.image_to_string(roi)
print(text)
