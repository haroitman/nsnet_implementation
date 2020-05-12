import librosa
import scipy
import numpy as np
from vad_utils import VadGenerator
from utils import wave_function

class AudioProcessing():
    
    def __init__(self, clean_speech_ordered_list, noisy_speech_ordered_list, noise_ordered_list, frames_per_sec, mode):
        self.clean_speech_ordered_list = clean_speech_ordered_list
        self.noisy_speech_ordered_list = noisy_speech_ordered_list
        self.noise_ordered_list = noise_ordered_list
        self.frames_per_sec = frames_per_sec
        self.mode = mode


    def frequency_independent_normalization(self, sample_number):
        # Normalize noisy speech, clean speech and noise audios based on noisy speech global mean and standard deviation 
        # frequency independent

        noisy_speech_zxx =  wave_function(self.noisy_speech_ordered_list[sample_number], mode=self.mode)
        noisy_speech_zxx_mean_fi = np.mean(noisy_speech_zxx)    
        noisy_speech_zxx_std_fi = np.std(noisy_speech_zxx)
        noisy_speech_normalized_fi = (noisy_speech_zxx - noisy_speech_zxx_mean_fi) / noisy_speech_zxx_std_fi

        noise_zxx = wave_function(self.noise_ordered_list[sample_number], mode='abs')
        noise_normalized_fi = noise_zxx

        clean_speech_zxx = wave_function(self.clean_speech_ordered_list[sample_number], mode='abs')
        clean_speech_normalized_fi = clean_speech_zxx

        return (noisy_speech_normalized_fi, clean_speech_normalized_fi, noise_normalized_fi)

    def adjusted_input_matrixes(self, sample_number, segment_frames_lenght):
        # Adjust the noisy speech matrix size to input it in the network
        # batch_size: number or audios in a training mini-batch
        # sample_number: number of the first audio sample used in the batch
        # segment_frames_lenght: number of past frames used for the training (160 if ) 

        noisy_speech_normalized_fi, clean_speech_normalized_fi, noise_normalized_fi = self.frequency_independent_normalization(sample_number)
        vad_gen = VadGenerator(self.clean_speech_ordered_list, self.frames_per_sec)
        vad_clean_speech_normalized = vad_gen.vad(sample_number) * clean_speech_normalized_fi

        number_of_frames = noisy_speech_normalized_fi.shape[1]
        dim1 = number_of_frames-segment_frames_lenght
        dim2 = segment_frames_lenght
        dim3 = noisy_speech_normalized_fi.shape[0]
        network_noisy_speech_input = np.zeros((dim1,dim2,dim3), float)
        network_clean_speech_input = np.zeros((dim1,dim2,dim3), float)
        network_vad_clean_speech_input = np.zeros((dim1,dim2,dim3), float)
        network_noise_input = np.zeros((dim1,dim2,dim3), float)
                
        for j in range(number_of_frames-segment_frames_lenght):
            for k in range(segment_frames_lenght):
                network_noisy_speech_input[j,k,:] = noisy_speech_normalized_fi[:,k+j]
                network_clean_speech_input[j,k,:] = clean_speech_normalized_fi[:,k+j]
                network_vad_clean_speech_input[j,k,:] = vad_clean_speech_normalized[:,k+j]
                network_noise_input[j,k,:] = noise_normalized_fi[:,k+j]

        return(network_noisy_speech_input,network_clean_speech_input, network_vad_clean_speech_input, network_noise_input)