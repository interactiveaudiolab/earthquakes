from scipy.signal import lfilter
import numpy as np
import librosa
from tqdm import trange
import h5py
import pickle
from utils import *

def extract_features(data, sr=40.0):
    raw_data = data / np.max(np.abs(data))
    b, a = butter_highpass(5.0, sr, 2)
    high_pass = lfilter(b, a, raw_data)

    b, a = butter_bandpass(2.0, 8.0, sr, 2)
    band_pass = lfilter(b, a, raw_data)

    b, a = butter_lowpass(2.0, sr, 2)
    low_pass = lfilter(b, a, raw_data)

    band_spectrogram = extract_spectrogram(band_pass, 2.0, 8.0)
    high_spectrogram = extract_spectrogram(high_pass, 5.0, 7.0)
    low_spectrogram = extract_spectrogram(low_pass, 0.0, 2.0)

    return np.vstack([band_spectrogram, low_spectrogram])[:, :50]


def extract_spectrogram(data, low_cutoff, high_cutoff, sr=40):
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(data, n_fft=1024, hop_length=512)), ref=0.0)
    fft_frequencies = librosa.fft_frequencies(sr=sr, n_fft=1024)
    low = (np.abs(fft_frequencies - low_cutoff)).argmin()
    high = (np.abs(fft_frequencies - high_cutoff)).argmin()
    # mel_filters = librosa.filters.mel(100, 2048, n_mels=300)
    # spectrogram = np.dot(mel_filters, spectrogram)
    return spectrogram[low:high, :]


earthquake_data = load_file('earthquake_data.p')

shape = (1,) + extract_features(positive_earthquakes[0]).shape

dataset = h5py.File('earthquake_data.h5', 'w')
dataset.create_group('training')
dataset.create_group('testing')

split = 'training'
amount_for_testing = positive_cutoff

dataset[split].create_dataset('data',
                              shape=shape,
                              maxshape=(None, None, None))
dataset[split].create_dataset('label',
                              shape=(1, 1),
                              maxshape=(None, None))

for i in trange(positive_cutoff):
    current_size = dataset[split]['data'].shape[0]
    dataset[split]['data'][current_size - 1] = extract_features(positive_earthquakes[i])
    dataset[split]['label'][current_size - 1] = 1
    dataset[split]['data'].resize(current_size + 1, axis=0)
    dataset[split]['label'].resize(current_size + 1, axis=0)

for i in trange(negative_cutoff):
    current_size = dataset[split]['data'].shape[0]
    dataset[split]['data'][current_size - 1] = extract_features(negative_earthquakes[i])
    dataset[split]['label'][current_size - 1] = 0
    if i < (len(negative_earthquakes) - 5) - 1:
        dataset[split]['data'].resize(current_size + 1, axis=0)
        dataset[split]['label'].resize(current_size + 1, axis=0)

split = 'testing'

dataset[split].create_dataset('data',
                              shape=shape,
                              maxshape=(None, None, None))
dataset[split].create_dataset('label',
                              shape=(1, 1),
                              maxshape=(None, None))
for i in tnrange(positive_cutoff, len(positive_earthquakes)):
    current_size = dataset[split]['data'].shape[0]
    dataset[split]['data'][current_size - 1] = extract_features(positive_earthquakes[i])
    dataset[split]['label'][current_size - 1] = 1
    dataset[split]['data'].resize(current_size + 1, axis=0)
    dataset[split]['label'].resize(current_size + 1, axis=0)

for i in tnrange(negative_cutoff, len(negative_earthquakes)):
    current_size = dataset[split]['data'].shape[0]
    dataset[split]['data'][current_size - 1] = extract_features(negative_earthquakes[i])
    dataset[split]['label'][current_size - 1] = 0
    if i < len(negative_earthquakes) - 1:
        dataset[split]['data'].resize(current_size + 1, axis=0)
        dataset[split]['label'].resize(current_size + 1, axis=0)