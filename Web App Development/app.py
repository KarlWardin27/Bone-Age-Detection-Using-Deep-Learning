#Authors:- Pratik Poojary, Prathamesh Pokhare

from flask import Flask, render_template, request
import tensorflow as tf
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from keras_preprocessing import image
import numpy as np
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

    img.save(secure_filename(img.filename))
    img = secure_filename(img.filename)

    model = tf.keras.models.load_model('best_model.h5', compile=False)

    img=tf.keras.preprocessing.image.load_img(img,target_size=(256,256))
    img=tf.keras.preprocessing.image.img_to_array(img)
    img=tf.keras.applications.xception.preprocess_input(img)

    #Calculated for the Model Training Part. Will vary with Training DataSet.
    Mean_Bone_Age = 127.3207517246848
    Standard_Bone_Age = 41.18202139939618

    Predict = round((mean_bone_age + std_bone_age*(model.predict(np.array([img]))[0][0]))/12, 2)

    return render_template('prediction.html', data = Predict)    

if __name__=='__main__':
    app.run(debug=True)   