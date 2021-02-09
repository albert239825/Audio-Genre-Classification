import streamlit as st
from pydub import AudioSegment
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import math

#convert the file into wav format
def convertToWav(input_audio):
    sound = AudioSegment.from_mp3(input_audio)
    if (len(sound) > 29000):
        print("success")
    else:   
        print("short file")
    song = sound.export(format = "wav")
    return song

#extract MFCCs
def extract_mfcc(signal):
    #variable declaration
    n_mfcc = 32
    n_fft = 2048
    hop_length = 512

    DURATION = len(signal) / SAMPLE_RATE
    SAMPLES_PER_TRACK = len(signal)
    #divide the song into 3 second segments so that we can feed into the model. Since it was trained on (129,32) data
    #segment can variable length??? Look into this...
    num_segments = (int) (len(signal) / (2.98 * SAMPLE_RATE))
    num_sam = int(SAMPLES_PER_TRACK / num_segments)

    expected_num_mfcc_vectors_per_segment = math.ceil(num_sam / hop_length) #round up
    print(expected_num_mfcc_vectors_per_segment)

    for s in range(num_segments):
        data = []
        start_sample = num_sam * s
        finish_sample = start_sample + num_sam

        mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], 
                                    sr = sr,
                                    n_fft = n_fft,
                                    n_mfcc = n_mfcc,
                                    hop_length = hop_length)
        mfcc = mfcc.T

        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
            data.append(mfcc.tolist())
            print("segment:{}".format(s+1))
        else:
            print("wrong length")
    return data




# st.write("""

# ## Audio Genre Classification

# Hello, this is a web app for audio genre classification

# """)



#getting the file
#uploaded_file = st.file_uploader("Upload Files", accept_multiple_files=True, type='mp3')

test_file = "Data/fma_small/000/000002.mp3"


#loading in the audiofile
SAMPLE_RATE = 22050
signal, sr = librosa.load(test_file, sr = SAMPLE_RATE)

data = extract_mfcc(signal)









# model = load_model('models/resnet50_pretrained_32.h5')

