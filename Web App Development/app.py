#Authors:- Pratik Poojary, Prathamesh Pokhare

from flask import Flask, render_template, request, url_for
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
 

@app.route('/index/', methods=['GET', 'POST'])
def index():

    img = request.files['image']

    img.save('static/'+secure_filename(img.filename))
    img_dir = 'static/' + secure_filename(img.filename)

    model = tf.keras.models.load_model('D:/Code_Me/Python/web_app/best_model.h5', compile=False)
    img=tf.keras.preprocessing.image.load_img(img_dir,target_size=(256,256))
    img=tf.keras.preprocessing.image.img_to_array(img)
    img=tf.keras.applications.xception.preprocess_input(img)
   
    mean_bone_age = 127.3207517246848
    std_bone_age = 41.18202139939618

    pred = round((mean_bone_age + std_bone_age*(model.predict(np.array([img]))[0][0]))/12, 2)

    final = list()
    final.append(pred)
    final.append(img_dir)

    return render_template('prediction.html', data = final)    

@app.route('/about/')
def about():
    return render_template('about.html')    

if __name__=='__main__':
    app.run(debug=True)   