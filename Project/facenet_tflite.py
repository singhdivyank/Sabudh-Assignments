import tensorflow as tf
from tensorflow.contrib import lite

converter = lite.TFLiteConverter.from_keras_model_file('facenet_keras.h5')
model = converter.convert()
file = open('model.tflite', 'wb')
file.write(model)
