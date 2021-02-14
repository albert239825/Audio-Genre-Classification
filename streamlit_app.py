import streamlit as st
from pydub import AudioSegment
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import math


#extract MFCCs
def extract_mfcc(signal, SAMPLE_RATE):
    #variable declaration
    n_mfcc = 32
    n_fft = 2048
    hop_length = 512

    #divide the song into 3 second segments so that we can feed into the model. Since it was trained on (129,32) data
    #Solving this my having fixed segment length. At most missing 3 seconds of the total song
    len_segment = (2.98 * SAMPLE_RATE)
    num_segments = (int) (len(signal) / len_segment)
    print("len_segment: {}, num_segments {}, cropped signal Length: {}, Original signal length: {}".format(len_segment, num_segments, (len_segment*num_segments), len(signal)))
    singal_cropped = signal[: (int) (len_segment*num_segments)]
    num_sam = int(len(singal_cropped) / num_segments)

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

data = extract_mfcc(signal, SAMPLE_RATE)


model = load_model('models/cnn_model_32.h5')

