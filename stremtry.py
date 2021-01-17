# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Copyright 2018-2020 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import io
import streamlit as st
import numpy as np
import wave
from scipy.io import wavfile
import sounddevice as sd
import soundfile as sf
import librosa
import IPython.display as ipd
import tensorflow
from tensorflow import keras
import time

st.title("Let's play with audio recognition!")
#st.image('audio_img.jpg')

# These are the formats supported in Streamlit right now.
AUDIO_EXTENSIONS = ["wav"]
samplerate = 16000
duration = 1
filename = 'try.wav'
classes = ['down','go','left','no','off','on','right','stop','up','yes']

# For samples of sounds in different formats, see
# https://docs.espressif.com/projects/esp-adf/en/latest/design-guide/audio-samples.html

model = keras.models.load_model('speech2text_model_21.hdf5')

def record():
    st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">Record your voice</p>',unsafe_allow_html=True)
    time.sleep(1.5)
    st.write('Start!')
    print("start")
    mydata=sd.rec(int(samplerate*duration),samplerate=samplerate,channels=1,blocking=True)
    print("end")
    sd.wait()
    #sf.write(filename,mydata,samplerate)
    #test,test_rate = librosa.load('try.wav',sr=1600)
    test_sample = librosa.resample(mydata,samplerate,8000)
    prob = model.predict(test_sample.reshape(1,8000,1))
    if np.amax(prob)<0.5:
        st.markdown("""
        <style>z
        .big-font {
            font-size:17px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<p class="big-font">I have not understood, can you repeat please?</p>',unsafe_allow_html=True)
        print('I have not understood, can you repeat please?')
        record()
    else:
        res = classes[np.argmax(prob)]
        print(classes[np.argmax(prob)])
        res = res.upper()
        st.write('You have said: ',res)
def get_audio_files_in_dir(directory):
    out = []
    for item in os.listdir(directory):
        try:
            name, ext = item.split(".")
        except:
            continue
        if name and ext:
            if ext in AUDIO_EXTENSIONS:
                out.append(item)
    return out

but = st.sidebar.button("Run")
if but:
    record()

listen = st.sidebar.button("Listen to your audio ")
if listen:
    avdir = os.path.expanduser("~")
    audiofiles = get_audio_files_in_dir(avdir)
    filename = st.selectbox(
        "Select an audio file from your home directory (%s) to play" % avdir,
        audiofiles,
        0,
    )
    audiopath = os.path.join(avdir, filename)
    st.audio(audiopath)






