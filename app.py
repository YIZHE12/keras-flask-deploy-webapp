from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model, Model
from keras.preprocessing import image
from keras.applications import vgg19

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from sklearn.preprocessing import normalize, scale
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.utils import Sequence
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model_exp_20_whole_classes_new_data_generator_focal_loss_f1-weights-improvement-24.hdf5'
index_label = ['Leggings', 'Long Sleeved', 'Male', 'Maternity', 'Mesh', 'Neutral',
       'Nylon', 'Off The Shoulder', 'Paisley', 'Party Dresses', 'Pleated',
       'Polka Dot', 'Polos', 'Polyester', 'Printed', 'Prom Dresses',
       'Quilted', 'Racerback', 'Rayon', 'Red', 'Bikinis', 'Ripped',
       'Round Neck', 'Ruched', 'Sequins', 'Short Sleeves', 'Black',
       'Sleeveless', 'Spaghetti Straps', 'Spandex', 'Blouses',
       'Strapless', 'Suits & Blazers', 'Summer', 'Sweetheart Neckline',
       'Swimsuits', 'Blue', 'T-Shirts', 'Tank Tops', 'Bodycon', 'Tulle',
       'Turtlenecks', 'U-Necks', 'V-Necks', 'Wedding Dresses', 'White',
       'Bubble Coats', 'Camouflage', 'Cargo Pants', 'Cargo Shorts',
       'Casual Dresses', 'Chiffon', 'Collared', 'Corsets', 'Cotton',
       'Denim', 'Dress Shirts', 'Dresses', 'Faux Fur', 'Female', 'Floral',
       'Formal Dresses', 'Furry', 'Galaxy', 'Gray', 'Green',
       'Hoodies & Sweatshirts', 'Jackets', 'Knit', 'Lace']
# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')

# load pre-trained VGG19 model for image embedding
base_model = vgg19.VGG19(weights='imagenet') #imports the mobilenet model
for layer in base_model.layers:
    layer.trainable=False
output = base_model.get_layer('fc2').output
model=Model(inputs=base_model.input, outputs=output)
model.compile('adam', 'binary_crossentropy')

# create the second model
n_class = 70
model2 = Sequential()
model2.add(Dense(1048, activation='relu', input_dim= 4096, kernel_initializer='he_normal'))
model2.add(BatchNormalization())
# Arch 3, add drop out
model2.add(Dropout(0.3))
model2.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
model2.add(Dropout(0.3))
model2.add(BatchNormalization())
# model.add(Dense(n_class, activation='softmax', kernel_initializer='he_normal'))
model2.add(Dense(n_class, activation='sigmoid', kernel_initializer='he_normal'))
model2.load_weights(MODEL_PATH)

pre_data = np.load('models/test_img_x.npy')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds

def model2_predict(preds, model2):
        preds2 = np.concatenate((preds,pre_data), axis = 0)
        # add other data to get the data distribution for normalization
        preds2 = normalize(preds2)
        preds2 = scale(preds2)
        results = model2.predict(preds2, batch_size = 5)
        return(results[0,:]) # only return the first sample which is the new sample



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        preds2 = model2_predict(preds, model2)

        index = np.where(preds2>0.4)
        index = np.asarray(index)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        # return result
        prediction_label = [index_label[i] for i in index[0]]
        return str(prediction_label) 
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
