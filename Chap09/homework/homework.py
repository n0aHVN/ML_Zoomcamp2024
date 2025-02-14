import tensorflow as tf
import tensorflow.lite as tflite
import os
import numpy as np
from tensorflow import keras
from io import BytesIO
from urllib import request
from PIL import Image
from keras.applications.xception import preprocess_input

print(np.__version__)

interpreter = tflite.Interpreter(model_path = "./model_2024_hairstyle.tflite")
interpreter.allocate_tensors()

output_index = interpreter.get_output_details()[0]['index']
input_index = interpreter.get_input_details()[0]['index']

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img
def prepare_input(x):
    return x / 255.0

def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(200, 200))

    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = prepare_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)
    print(preds)
    float_predictions = preds[0].tolist()
    return float_predictions

def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)
    result = {
        'prediction': pred
    }

    return result
