import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS  # Import Flask-CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Initialize VGG19 model
base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(base_model.inputs, output)
model_03.load_weights('vgg_unfrozen.h5')

app = Flask(__name__)
CORS(app)  # Enable CORS

print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
    return "Yes Brain Tumor" if classNo == 1 else "No Brain Tumor"


def getResult(img):
    try:
        image = cv2.imread(img)
        if image is None:
            raise ValueError(f"Failed to read image: {img}")

        image = Image.fromarray(image, 'RGB')
        image = image.resize((240, 240))
        image = np.array(image)
        input_img = np.expand_dims(image, axis=0)
        result = model_03.predict(input_img)
        result01 = np.argmax(result, axis=1)
        return result01[0]
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        value = getResult(file_path)
        if value is not None:
            result = get_className(value)
        else:
            result = "Error processing the image."

        print(f"Returning result: {result}")
        return jsonify(result=result)
    return None


if __name__ == '__main__':
    app.run(debug=True)
