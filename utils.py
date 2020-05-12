import librosa
import scipy
import numpy as np

def wave_function(audio, mode = 'log'):
    # generate the wave of the audio sample
    wave, sr = librosa.load(audio, sr = 16000, mono = True)
    f, t, stft  = scipy.signal.stft(wave, fs=16000, window='hamming', nperseg=512, noverlap=0.75)
    if mode == 'log':
        lps = np.log(np.maximum(np.square(np.absolute(stft)), 10e-12))
    elif mode == 'abs':
        lps = np.absolute(stft)
    else:
        lps = stft
    return lps
