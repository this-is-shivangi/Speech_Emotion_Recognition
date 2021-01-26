# Speech-Emotion-Recognition

This repository contains work on speech emotion recognition using ravdess dataset.

### Installing dependencies
*Note*: You can skip this step, if you are installing the packages. 
Dependencies are listed below and in the `requirements.txt` file.

* scipy
* sklearn
* speechrecognition
* pyaudio
* keras
* tensorflow
* librosa
* seaborn
* numpy
* os
* glob

Install one of python package managers in your distro. If you install pip, then you can install the dependencies by running 
`pip3 install -r requirements.txt` 

If you prefer to accelerate keras training on GPU's you can install `tensorflow-gpu` by 
`pip3 install tensorflow-gpu`

### Directory Sturcture
- `speech-emotion-recognition-ravdess/` -  Contains the speech files in wav formatted seperated into 12 folders which are the corresponding labels of those files
- `clean_audio/` - Contains all the masked files
- `user_audio/` - Contains the real time of the user


- `ml_models/speech.py` - Contains the code for non-neural model
- `ml_models/dnn/neural.py` - Contains the code for neural model.
