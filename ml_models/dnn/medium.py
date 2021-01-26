import glob
import os
import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import confusion_matrix


def extract_feature(file_name):
    # print("feature extarct", file_name)
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


def parse_audio_files(parent_dir, sub_dirs, file_ext="*.wav"):
    print("Parsing files")
    features, labels = np.empty((0, 193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        print(parent_dir, "/" , sub_dir)
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))):
            try:
                mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            except Exception as e:
                print("Error encountered while parsing file: ", fn)
                continue
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            features = np.vstack([features, ext_features])
            labels = np.append(labels, fn.split('\\')[-1].split('-')[2])
    return np.array(features), np.array(labels, dtype=np.int)


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels+1))
    one_hot_encode[np.arange(n_labels), labels] = 1
    one_hot_encode = np.delete(one_hot_encode, 0, axis=1)
    return one_hot_encode


main_dir = r'C:\Users\shivangi shukla\Documents\Projects\SER\speech-emotion-recognition-ravdess-data'
sub_dir = os.listdir(main_dir)
print("\ncollecting features and labels...")
print("\nthis will take less than 10 minutes...")
features, labels = parse_audio_files(main_dir, sub_dir)
print("done")
np.save('X', features)
# one hot encoding labels
labels = one_hot_encode(labels)
np.save('y', labels)

X = np.load('X.npy')
y = np.load('y.npy')
train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.33, random_state=42)

n_dim = train_x.shape[1]
n_classes = train_y.shape[1]
n_hidden_units_1 = n_dim
n_hidden_units_2 = 400  # approx n_dim * 2
n_hidden_units_3 = 200  # half of layer 2
n_hidden_units_4 = 100

# defining the model


def create_model(activation_function='relu', init_type='random_normal', optimiser='adam', dropout_rate=0.2):
    model = Sequential()
    # layer 1
    model.add(Dense(n_hidden_units_1, input_dim=n_dim,
                    kernel_initializer=init_type, activation=activation_function))
    # layer 2
    model.add(Dense(n_hidden_units_2, kernel_initializer=init_type,
                    activation=activation_function))
    model.add(Dropout(dropout_rate))
    # layer 3
    model.add(Dense(n_hidden_units_3, kernel_initializer=init_type,
                    activation=activation_function))
    model.add(Dropout(dropout_rate))
    # layer4
    model.add(Dense(n_hidden_units_4, kernel_initializer=init_type,
                    activation=activation_function))
    model.add(Dropout(dropout_rate))
    # output layer
    model.add(Dense(n_classes, kernel_initializer=init_type, activation='softmax'))
    # model compilation
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimiser, metrics=['accuracy'])
    return model


# create the model
model = create_model()
# train the model
history = model.fit(train_x, train_y, epochs=200, batch_size=4)

# predicting from the model
predict = model.predict(test_x, batch_size=4)
model.save("model")

# converting numbers to emotions
emotions = ['neutral', 'calm', 'happy', 'sad',
            'angry', 'fearful', 'disgust', 'surprised']
# predicted emotions from the test set
y_pred = np.argmax(predict, 1)
predicted_emo = []
for i in range(0, test_y.shape[0]):
    emo = emotions[y_pred[i]]
    predicted_emo.append(emo)
# print("predicted values", predicted_emo)
actual_emo = []
y_true = np.argmax(test_y, 1)
for i in range(0, test_y.shape[0]):
    emo = emotions[y_true[i]]
    actual_emo.append(emo)

#print("actual        :       predicted values")
# for _ in range(len(actual_emo)):
#    print(actual_emo[_], ":", predicted_emo[_])

# generate the confusion matrix
# cm =confusion_matrix(actual_emo, predicted_emo)
# index = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
# columns = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
# cm_df = pd.DataFrame(cm,index,columns)
# plt.figure(figsize=(10,6))
# sns.heatmap(cm_df, annot=True)

# model = load_model("model")
# file = 'user_audio/final.wav'
# new_features = np.empty((0, 193))
# try:
#     n_mfccs, n_chroma, n_mel, n_contrast, n_ton
# netz = extract_feature(file)
# except Exception as e:
#     print("Error encountered while parsing file: ", file)

# ext_features = np.hstack([n_mfccs, n_chroma, n_mel, n_contrast, n_tonnetz])
# new_features = np.vstack([new_features, ext_features])

# # print("ans", new_features)
# # print("test", test_x)

# pred = model.predict(new_features)
# n_pred = np.argmax(pred, 1)
# print("Predicted Emotion")
# for i in n_pred:
#     print(emotions[i])
