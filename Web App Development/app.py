from flask import Flask, render_template, request
import tensorflow as tf
from keras_preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
import threading
import time
import sys


app=Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')
 

@app.route('/home', methods=['POST'])
def home():
    
    img = request.files['image']

    model = tf.keras.models.load_model('best_model.h5', compile=False)
    #prediction = model.predict(img_arr)
    img=tf.keras.preprocessing.image.load_img(img,target_size=(256,256))

    img=tf.keras.preprocessing.image.img_to_array(img)
    img=tf.keras.applications.xception.preprocess_input(img)
    mean_bone_age = 127.3207517246848
    std_bone_age = 41.18202139939618

    pred = mean_bone_age + std_bone_age*(model.predict(np.array([img])))
 
    '''x = round(prediction[0,0], 2)
    y = round(prediction[0,1], 2)
    preds = np.array([x,y])
    COUNT += 1'''
    return render_template('prediction.html', data=pred)    

if __name__=='__main__':
    app.run(debug=True)   