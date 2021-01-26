import wave
import pyaudio
from array import array
import librosa
import soundfile
import pandas as pd
import os
import glob
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
# from soundfile import Sample_rate
from tqdm import tqdm
from scipy.io import wavfile

# vars
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprise'
}
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# functions


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X, sample_rate = librosa.load(file_name)
        # X = sound_file.read(dtype="float32")
        # sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,
                                                 n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft,
                                                         sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X,
                                                         sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result


def load_data(test_size=0.33):
    print("Loading Data")
    x, y = [], []
    for file in tqdm(glob.glob(r'C:\Users\shivangi shukla\Documents\Projects\minor project\speech-emotion-recognition-ravdess-data\\**\\*.wav')):
        file_name = os.path.basename(file)
        signal, rate = librosa.load(file, sr=16000)
        mask = envelope(signal, rate, 0.0005)
        clean_file_name = r'C:\Users\shivangi shukla\Documents\Projects\minor project\clean_audio\\'+str(file_name)
        wavfile.write(filename=r'C:\Users\shivangi shukla\Documents\Projects\minor project\clean_audio\\'+str(file_name),
                      rate=rate, data=signal[mask])
        emotion = emotions[clean_file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append([emotion, file_name])
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


x_train, x_test, y_train, y_test = load_data(test_size=0.25)

print(np.shape(x_train), np.shape(x_test), np.shape(y_train), np.shape(y_test))

y_test_map = np.array(y_test).T
y_test = y_test_map[0]
test_filename = y_test_map[1]
y_train_map = np.array(y_train).T
y_train = y_train_map[0]
train_filename = y_train_map[1]

# print(np.shape(y_train),np.shape(y_test))
# print(*test_filename,sep="\n")

print(f'Features extracted:{x_train.shape[1]}')

model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08,
                      hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_pred = pd.Series(y_pred)
y_pred

accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

y_pred1 = pd.DataFrame(y_pred, columns=['preedictions'])
y_pred1['file_names'] = test_filename
y_pred1.to_csv('prediction.csv')

Pkl_Filename = "Speech_Emotion_Recognition_Model.pkl"

with open(Pkl_Filename, 'wb') as file:
    pickle.dump(model, file)

with open(Pkl_Filename, 'rb') as file:
    speech_recognition = pickle.load(file)

speech_recognition


print("Starting voice record...")
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 6
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
frames = []

check = 0

for i in tqdm(range(0, int(RATE/CHUNK * RECORD_SECONDS))):
    data = stream.read(CHUNK)
    data_chunk = array('h', data)
    vol = max(data_chunk)
    if(vol >= 500):
        check = 1
        frames.append(data)

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

if(check == 1):
    data,  sampling_rate = librosa.load(WAVE_OUTPUT_FILENAME)
    # %matplotlib inline
    import matplotlib.pyplot as plt
    import librosa.display

    plt.figure(figsize=(15, 5))
    librosa.display.waveplot(data, sr=sampling_rate)

    file = WAVE_OUTPUT_FILENAME
    ans = []
    new_feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
    ans.append(new_feature)
    ans = np.array(ans)

    # Speech_Emotion_Recognition_Model.predict([ans])
    pred = model.predict(ans)
    pred = pd.Series(pred)
    print(pred[0])
else:
    print("Nothing Recorded")
