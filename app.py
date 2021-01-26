import os
from datetime import datetime
import glob
from utils import *
import sqlalchemy
from flask import (Flask, abort, flash, redirect, render_template, request,
                   session, url_for)
from sqlalchemy import create_engine, engine
from sqlalchemy.orm import scoped_session, sessionmaker
from werkzeug.utils import \
    secure_filename                                             # it helps to covert bad filename into a secure filename
from project_orm import Records
from tensorflow import keras
import numpy as np
import librosa
import pyaudio
from array import array
from tqdm import tqdm
import wave
from scipy.io import wavfile

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 2                                                       # 2MB
app.config['UPLOAD_EXTENSIONS'] = ['.wav','.mp4','.mp3','.ogg']
app.config['UPLOAD_PATH' ] = 'user_audio'
app.secret_key='no error message'


@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        f = request.files['audio_data']
        file = "./user_audio/audio.wav"
        with open(file , 'wb') as audio:
            f.save(audio)
        print('file uploaded successfully')

        return render_template('index.html', request="POST")
    else:
        return render_template("index.html")

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(
        S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(
        S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


@app.route('/result' , methods = ["POST"])
def prediction():
    if request.method == 'POST':
        model = keras.models.load_model("model")
        list_of_files = glob.glob('./user_audio/*.wav') # * means all if need specific format then *.csv
        file = max(list_of_files, key=os.path.getctime)
        new_features = np.empty((0, 193))
        emotions = ['neutral', 'calm', 'happy', 'sad',
            'angry', 'fearful', 'disgust', 'surprised']

        n_mfccs, n_chroma, n_mel, n_contrast, n_tonnetz = extract_feature(file)
        ext_features = np.hstack([n_mfccs, n_chroma, n_mel, n_contrast, n_tonnetz])
        new_features = np.vstack([new_features, ext_features])
        pred = model.predict(new_features)
        n_pred = np.argmax(pred, 1)
        for i in n_pred:
            result = emotions[i]
    return render_template('index.html' , result=result)

if __name__ == "__main__":
    app.run(debug=True)