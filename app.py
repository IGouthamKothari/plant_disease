import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model.h5', compile=False)
print('Model loaded.')


def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(224, 224))  # Adjust image size for mobile devices
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(file.filename))
    file.save(file_path)

    # Make prediction
    preds = model_predict(file_path, model)
    print(preds[0])

    disease_class = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
                     'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                     'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                     'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                     'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
    a = preds[0]
    ind = np.argmax(a)
    print('Prediction:', disease_class[ind])
    result = disease_class[ind]
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True)
