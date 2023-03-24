import os
from flask import Flask, render_template, request, jsonify, url_for, send_file, send_from_directory
from keras.utils import load_img
from keras.utils import img_to_array
import tensorflow as tf
import numpy as np
from PIL import Image
import shutil

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './img/'



target_names = ['COVID','Normal','Viral Pneumonia']

def load_model():
    model = tf.keras.models.load_model('./model_vgg.h5')
    model.compile(
        loss='binary_crossentropy',
        optimizer='SGD',
        metrics=['accuracy']
    )
    return model

def is_grey_scale(img_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i,j))
            if r != g != b: 
                return False
    return True

@app.route('/', methods = ['GET'])
def covid_detection():
    shutil.rmtree('./img/')
    os.mkdir('./img/')
    return render_template("index.html")

@app.route('/', methods = ['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224,224))
    if (is_grey_scale(image_path) == True):
        image1 = img_to_array(image)
        image1 = np.expand_dims(image1, axis=0)
        image1 = np.vstack([image1])
        model = load_model()
        prediksi = model.predict(image1)
        skor = np.max(prediksi)
        classes = np.argmax(prediksi)
        if skor > 0.9:
            hasil = target_names[classes]
        else:
            hasil='Tidak terdeteksi apapun, periksa gambar Anda'
    else:
        hasil = 'Gambar tidak terdeteksi sebagai citra x-ray'

    return render_template("hasil.html", result=hasil, img=imagefile.filename)

@app.route('/img/<fileimg>')
def send_uploaded_image(fileimg=''):
    return send_from_directory( app.config['UPLOAD_FOLDER'], fileimg)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
