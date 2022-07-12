import os
from flask import Flask, render_template, request, jsonify, url_for
from keras.utils import load_img
from keras.utils import img_to_array
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model('./model_vgg.h5')
model.compile(
    loss='binary_crossentropy',
    optimizer='SGD',
    metrics=['accuracy']
)

target_names = ['COVID-19','Normal','Viral Pneumonia']

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
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = './img/' + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224,224))
    if (is_grey_scale(image_path) == True):
        # image = img_to_array(image)
        image = np.float32(image)/255
        image = np.expand_dims(image, axis=0)
        # image = np.vstack([image])
        prediksi = model.predict(image)
        skor = np.max(prediksi)
        print(skor)
        classes = np.argmax(prediksi)
        if skor > 0.9:
            hasil = target_names[classes]
        else:
            hasil='Tidak terdeteksi apapun, periksa gambar Anda'
    else:
        hasil = 'Gambar tidak terdeteksi sebagai citra x-ray'

    os.remove(image_path)
    return render_template("hasil.html", result=hasil)

if __name__ == '__main__':
    app.run(port=3000, debug=True)