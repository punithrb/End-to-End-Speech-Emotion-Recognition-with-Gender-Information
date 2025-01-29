from flask import Flask, render_template, request, url_for, redirect, flash, send_from_directory, session
from forms import RegistrationForm, LoginForm
# import pymysql
import mysql.connector
import pandas as pd
import os
from audio_wave import *
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import librosa
import IPython.display as ipd
import numpy as np
import pandas as pd

from model import get_gender

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
mydb = mysql.connector.connect(
    host="localhost", user="root", passwd="", port = "3307", database="emotion_recognition")
cursor = mydb.cursor()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(APP_ROOT, 'static/image/')
app.config['SECRET_KEY'] = 'b0b4fbefdc48be27a6123605f02b6b86'

model_name_or_path = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"

# audio_paths = ["/kaggle/input/speech-emotion-recognition-en/Ravdess/audio_speech_actors_01-24/Actor_12/03-01-02-01-02-01-12.wav"] # Must be a list with absolute paths of the audios that will be used in inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label2id = {
    "female": 0,
    "male": 1
}

id2label = {
    0: "female",
    1: "male"
}

num_labels = 2

def initialize():
    session['loggedin'] = False


@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/application')
def application():
    return render_template('application.html')


@app.route('/contact', methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        return render_template('contact.html', message ="Message send successfully! We will contact you very soon!")
    return render_template('contact.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":

        email = request.form['email']

        password1 = request.form['pwd']

        sql = "select * from register where email='%s' and pwd='%s' " % (
            email, password1)
        print('q')
        x = cursor.execute(sql)
        print(x)
        results = cursor.fetchall()

        if len(results) > 0:
            name = results[0][1]
            flash("Welcome to website", "primary")
            return render_template('application.html', m="Login Success", msg=results[0][1])

        else:
            flash("Login failed", "warning")
            return render_template('login.html', msg="Login Failure!!!")
    return render_template('login.html')


@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        pwd = request.form['pwd']
        cpwd = request.form['cpwd']

        sql = "select * from register"
        result = pd.read_sql_query(sql, mydb)
        email1 = result['email'].values
        print(email1)
        if email in email1:
            flash("email already existed", "warning")
            return render_template('register.html', msg="email existed")
        if (pwd == cpwd):
            sql = "INSERT INTO register (name,email,pwd,cpwd) VALUES (%s,%s,%s,%s)"
            val = (name, email, pwd, cpwd)
            cursor.execute(sql, val)
            mydb.commit()
            flash("Successfully Registered", "warning")
            return render_template('login.html')
        else:
            flash("Password and Confirm Password not same", '')
        return render_template('register.html')

    return render_template('register.html')


# Define function to extract MFCC features from a single audio file
def extract_mfcc(file_path, n_mfcc=13):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)  # Take mean of each MFCC feature across time
    return mfccs_mean

def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)  # Ensure correct resampling
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def predict(path, sampling_rate):
    # Convert audio file to array
    speech = speech_file_to_array_fn(path, sampling_rate)
    
    # Extract features from the audio file
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}

    # Predict using the model
    with torch.no_grad():
        logits = model(**inputs).logits

    # Apply softmax to get the scores
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    
    # Map short labels ('F' and 'M') to full labels ('Female', 'Male')
    label_mapping = {'F': 'Female', 'M': 'Male'}
    max_index = scores.argmax()
    predicted_label = label_mapping.get(config.id2label[max_index], config.id2label[max_index])
    
    return predicted_label


@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/audio/')

    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files["myaudio"]
    filename = file.filename
    if filename == "":
        flash('No File Selected', 'danger')
        return redirect(url_for('application'))

    destination = "/".join([target, filename])
    # Extension check
    ext = os.path.splitext(destination)[1]
    if (ext == ".wav"):
        pass
    else:
        flash("Invalid Extenstions! Please select a .wav audio file only.",
              category="danger")
        return redirect(url_for('application'))

    if not os.path.isfile(destination):
        file.save(destination)

    result = record_audio(record=False, file_loc=destination)
    new_dest = ('static/audio/'+str(filename))
    # Load the saved model
    model_filename = 'gender_classification_model.joblib'
    rf_model = joblib.load(model_filename)
    print("Model loaded successfully.")

    # Process the audio file for emotion detection (or any other audio analysis)
    emotion_result = record_audio(record=False, file_loc=destination)

    # Load the pre-trained gender classification model
    model_filename = 'gender_classification_model.joblib'
    rf_model = joblib.load(model_filename)
    print("Model loaded successfully.")

    # Load the label encoder for the gender labels
    label_encoder_filename = 'label_encoder.pkl'
    label_encoder = joblib.load(label_encoder_filename)

    # Extract MFCC features from the audio file for gender prediction
    features = extract_mfcc(destination)
    features_reshaped = features.reshape(1, -1)  # Reshape to match model input

    # Predict gender
    predicted_class = rf_model.predict(features_reshaped)
    predicted_label = label_encoder.inverse_transform(predicted_class)[0]
    audio_paths=[destination]
    print(11111111,audio_paths)
    
    predicted_class = get_gender(model_name_or_path, audio_paths, label2id, id2label, device)
    print(5555555,predicted_class)
    labels=["female","male"]
    # Map the predicted indices to the corresponding labels
    predicted_labels = [labels[pred] for pred in predicted_class]

    # If you expect only one prediction, you can get the first label directly
    predicted_labels = predicted_labels[0]  
    print(f"The predicted gender for the audio file '{filename}' is: {predicted_labels}")

    # Render the upload template with results
    return render_template(
        "upload.html", 
        img_name=filename, 
        emo=emotion_result, 
        destination=os.path.join('static/audio', filename),
        gender=predicted_labels
    )


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    # flash('Logged Out Sucessfully!','success')
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
