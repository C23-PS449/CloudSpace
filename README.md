# CloudSpace
This is Repository for Cloud in this Capstone Project

Explanation of back-end and cloud computing

# main.py 
Is an API implementation using the Flask framework to predict labels on images in our group. Here is an explanation for each part of the code:

#Import library
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

In this section, we set the TF_CPP_MIN_LOG_LEVEL environment variable so that TensorFlow only displays important log messages (only ERROR log messages).

import io
import tensorflow
from tensorflow import keras
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

This section is for importing the necessary libraries in the application, such as io for input/output operations, tensorflow and keras for modeling and prediction using pre-trained models, numpy for numerical operations, PIL for manipulating images, and Flask for building APIs.

#Load model
model = keras.models.load_model("RiceBuddy_model.h5")

This line loads the trained model from "RiceBuddy_model.h5" file using Keras. This model will be used to make predictions on uploaded images.

#List of class labels
label = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]

The list of class labels is used to display the predictive labels provided by the model.

#Initialize flask
app = Flask(__name__)

Initialize the Flask object as the main application.


#Label prediction function
def predict_label(img):
   i = np.asarray(img) / 255.0
   i = i.reshape(1, 100, 100, 3)
   pred = model.predict(i)
   result = label[np.argmax(pred)]
   return result

The predict_label function accepts an image as input, pre-processes the image, and uses the model to make predictions. The prediction results are converted to class labels using the argmax index on the model output.

#Prediction endpoints
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

This is the /predict endpoint that accepts GET or POST methods. When an image is submitted via a POST request, it is uploaded, resized to 100x100 pixels, and then predicted using the predict_label function. The prediction label is then returned as a response from the API.

#Running applications
if __name__ == "__main__":
   app.run(debug=True)

This is the part that runs the Flask application in debug mode.

In essence the code builds a simple API that accepts an image via a POST request on the /predict endpoint and returns the class label prediction for that image using the trained model.


# Dockerfile
Used to build Docker images for Python applications. Here is an explanation for each part of the code:

#Use the official lightweight Python image.
FROM python:3.9-slim

This line instructs Docker to use the official 3.9-slim lightweight Python image as the basis for building the image.

#Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

This line sets the PYTHONUNBUFFERED environment variable as True. This allows statements and log messages from Python applications to be directly displayed in Knative logs.

#Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

This section sets the APP_HOME environment variable as /app and changes the working directory to /app. Next, all files and directories from the current local directory will be copied into the Docker image in the /app directory.

#Install production dependencies.
RUN pip install -r requirements.txt

This command runs pip install to install all production dependencies defined in the requirements.txt file inside the Docker image.

#Run the web service on container startup. Here we use the gunicorn
#webserver, with one worker process and 8 threads.
#For environments with multiple CPU cores, increase the number of workers
#to be equal to the cores available.
#Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app

This line defines the command to be executed when the container is started. Here, we are using the gunicorn web server to run the application. --bind :$PORT binds the application to the port provided by the $PORT environment variable. --workers 1 sets the number of worker processes to one. --threads 8 sets the number of threads per worker process to eight. --timeout 0 disables timeouts for workers so that Cloud Run can handle instance scaling. main:app refers to the main.py module and the app object that gunicorn will run. This Dockerfile instructs Docker to build the image containing the Python application, install dependencies, and run the application using gunicorn when the container is started.


# requirements.txt
Contains all libraries imported in the API and ML models, with the addition of the gunicorn library. Gunicorn is a HTTP WSGI (Web Server Gateway Interface) web server for Python. It is used to run Python applications in a production environment. Gunicorn manages HTTP connections, organizes workers to handle requests, and provides good performance against high web traffic. Must contain gunicorn, otherwise the code will error.


# cloudbuild.yaml
is a configuration file used by Cloud Build, Google Cloud Platform's CI/CD (Continuous Integration/Continuous Deployment) service. This file is used to define the steps that Cloud Build must take when building, testing, and deploying your application or infrastructure. Here's a simple example of a function that would normally be in a cloudbuild.yaml file:

code:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/[PROJECT_ID]/[IMAGE_NAME]', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/[PROJECT_ID]/[IMAGE_NAME]']

In the code above, there are two steps that will be executed by Cloud Build:
The first step (- name: 'gcr.io/cloud-builders/docker') uses the Docker builder to build the Docker image from the current project directory (.) using the docker build command. The Docker image is then tagged with gcr.io/[PROJECT_ID]/[IMAGE_NAME] (in the image name format used by the Google Container Registry).

The second step (- name: 'gcr.io/cloud-builders/docker') uses the Docker builder to upload (push) the built Docker image to the Google Container Registry. The Docker image will be uploaded to gcr.io/[PROJECT_ID]/[IMAGE_NAME].

The cloudbuild.yaml file can include other steps, such as running test scripts, deploying infrastructure using a tool such as Terraform, or performing other custom steps that match the needs of your application or infrastructure. It's important to note that the actual content of cloudbuild.yaml will depend heavily on your project needs and configuration. You can customize the file to meet the required steps in your project's CI/CD cycle using Cloud Build.


# .gitignore
.gitignore file is a file used by Git to ignore files or directories it doesn't want followed and included in a Git repository. This file contains patterns used to specify which files or directories Git should ignore.

# .dockerignore
.dockerignore file is a file used by Docker to ignore files or directories that don't need to be included in the Docker image build process. This file is similar to a .gitignore file, but specifically for use with Docker.

We also need to inform you that we are deploying the ML model and ML API on cloud run.
