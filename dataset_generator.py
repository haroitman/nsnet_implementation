import numpy as np
from audio_processing import AudioProcessing
from tensorflow.keras import utils

class DatasetGenerator(utils.Sequence) :
    
    def __init__(self, clean_files, noisy_files, noise_files, frames_per_sec, batch_size):
        self.clean_files = clean_files
        self.noisy_files = noisy_files
        self.noise_files = noise_files
        self.frames_per_sec = frames_per_sec
        self.batch_size = batch_size
        
    def __len__(self) :
        return (np.ceil(len(self.noisy_files) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx) :
        max_samples = 1000
        window_lenght = 32
        number_features = 257
        noisy = np.zeros((self.batch_size*max_samples, window_lenght, number_features), float)
        clean = np.zeros((self.batch_size*max_samples, window_lenght, number_features), float)
        clean_vad = np.zeros((self.batch_size*max_samples, window_lenght, number_features), float)
        noise = np.zeros((self.batch_size*max_samples, window_lenght, number_features), float)
        pointer = 0

        audio_processing = AudioProcessing(self.clean_files, self.noisy_files, self.noise_files, self.frames_per_sec, 'abs')
        for i in range(idx*self.batch_size, (idx+1)*self.batch_size):
            noisy_item, clean_item, clean_vad_item, noise_item = audio_processing.adjusted_input_matrixes(i, window_lenght)
            noisy[pointer:pointer+noisy_item.shape[0],:,:] = noisy_item
            clean[pointer:pointer+clean_item.shape[0],:,:] = clean_item
            clean_vad[pointer:pointer+clean_vad_item.shape[0],:,:] = clean_vad_item
            noise[pointer:pointer+noise_item.shape[0],:,:] = noise_item
            pointer = pointer + min(noisy_item.shape[0], clean_item.shape[0], clean_vad_item.shape[0], noise_item.shape[0])
        noisy = noisy[0:pointer, :, :]
        clean = clean[0:pointer, :, :]
        clean_vad = clean_vad[0:pointer, :, :]
        noise = noise[0:pointer, :, :]
        
        clean = clean[:,-1,:].reshape((clean.shape[0], clean.shape[2]))
        clean_vad = clean_vad[:,-1,:].reshape((clean_vad.shape[0], clean_vad.shape[2]))
        noise = noise[:,-1,:].reshape((noise.shape[0], noise.shape[2]))
        
        samples_per_batch = self.batch_size * 280
        noisy = noisy[:samples_per_batch,:,:]
        clean = clean[:samples_per_batch,:]
        clean_vad = clean_vad[:samples_per_batch,:]
        noise = noise[:samples_per_batch,:]
        
        return [noisy, clean, clean_vad, noise]

