import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import tensorflow
from tensorflow import keras
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

model = keras.models.load_model("RiceBuddy_model.h5")
# Daftar label kelas
label = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]

app = Flask(__name__)

#Function yang berfungsi untuk melakukan prediksi pada gambar yang diinput
def predict_label(img):
   i = np.asarray(img) / 255.0
   #sesuai dengan input_size yang digunakan pada saat pembuatan model
   i = i.reshape(1, 100, 100, 3)
   pred = model.predict(i)
   result = label[np.argmax(pred)]
   return result

@app.route("/predict", methods=["GET","POST"])
def index():
   file = request.files.get('file')
   if file is None or file.filename == "":
         return jsonify({"error": "no file"})

   image_bytes = file.read()
   img = Image.open(io.BytesIO(image_bytes))
   img = img.resize((100,100), Image.NEAREST)
   pred_img = predict_label(img)
   return pred_img

if __name__ == "__main__":
    app.run(debug=True)