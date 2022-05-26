import streamlit as st
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import math
import numpy as np
import pickle


# extract MFCCs
def extract_mfcc(signal, SAMPLE_RATE):
    # variable declaration
    n_mfcc = 32
    n_fft = 2048
    hop_length = 512

    # divide the song into 3 second segments so that we can feed into the model. Since it was trained on (129,32) data
    # Solving this my having fixed segment length. At most missing 3 seconds of the total song
    len_segment = (2.98 * SAMPLE_RATE)
    num_segments = (int)(len(signal) / len_segment)
    print("len_segment: {}, num_segments {}, cropped signal Length: {}, Original signal length: {}".format(
        len_segment, num_segments, (len_segment*num_segments), len(signal)))
    singal_cropped = signal[: (int)(len_segment*num_segments)]
    num_sam = int(len(singal_cropped) / num_segments)

    expected_num_mfcc_vectors_per_segment = math.ceil(
        num_sam / hop_length)  # round up
    print(expected_num_mfcc_vectors_per_segment)

    st.write("processing Audio")

    data = []
    for s in range(num_segments):
        start_sample = num_sam * s
        finish_sample = start_sample + num_sam

        mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                    sr=sr,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length)
        mfcc = mfcc.T

        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
            data.append(mfcc.tolist())
            print("segment:{}".format(s+1))
        else:
            print("wrong length")

    data = np.array(data)
    return data


st.write("""

## Audio Genre Classification

Hello, this is a web app for audio genre classification

""")


test_file = "Data/fma_small/000/000002.mp3"

with open('data/mapping.pickle', 'rb') as f:
    mapping = pickle.load(f)

# getting the file
uploaded_file = st.file_uploader("Upload Files", type='.mp3')

# checking that we have an uploaded file
if uploaded_file is not None:
    # note for later, how might this conflict if there are two people running the same script
    filepath = 'temp/' + uploaded_file.name
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # display the audio file
    st.audio(filepath, format='audio/mp3')

    # loading in the audiofile
    SAMPLE_RATE = 22050
    signal, sr = librosa.load(filepath, sr=SAMPLE_RATE)

    data = extract_mfcc(signal, SAMPLE_RATE)
    # expanding dimension to feed into CNN
    data_cnn = np.expand_dims(data, axis=-1)

    model = load_model('models/cnn_model_32.h5')
    predictions = model.predict(data_cnn)
    # used to calculate how sure we are
    total_prob = predictions.shape[0]
    sum_predictions = predictions.sum(axis=0)
    predicted_indicies = np.argsort(sum_predictions)

    print(sum_predictions, predicted_indicies, mapping)

    # Really Messy Print statement that tells us the prediction certainties
    st.write("The first prediction is: {} with a {}% certainty , the second guess is: {} with a {}% certainty"
             .format(mapping[predicted_indicies[-1]], round((sum_predictions[predicted_indicies[-1]] / total_prob) * 100, 2),
                     mapping[predicted_indicies[-2]], round((sum_predictions[predicted_indicies[-2]] / total_prob) * 100, 2)))

    os.remove(filepath)
