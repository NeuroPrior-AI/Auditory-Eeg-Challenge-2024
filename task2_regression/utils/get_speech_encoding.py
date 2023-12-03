
import os
import numpy as np
import sys
sys.path.append('/home/naturaldx/auditory-eeg-challenge-2024-code')

from task2_regression.models.SpeechAutoEncoder import SpeechAutoEncoder
from task2_regression.configs.SAE import model_config, train_config
from util.data_loader import *
from util.wav2vec2 import mel_to_audio, speech_encoder


train_loader, val_loader = create_train_val_loader(batch_size=64)

def save_data(eegs, mels, encodings, file_path):
    """ Save EEG, original mels, and encodings to a file """
    np.savez(file_path, eegs=eegs.cpu().numpy(), mels=mels.cpu().numpy(), encodings=encodings.cpu().numpy())

def process_and_save_data(train_loader, save_dir):
    """ Process and save the data """
    for i, (eegs, mels) in enumerate(train_loader):
        waveforms = [mel_to_audio(mel) for mel in mels]
        latents = speech_encoder(waveforms, sampling_rate=model_config['sampling_rate'])
        # latents = latents.transpose(1, 2)  # Transpose to (batch_size, latent_dim, length)
        print("[DIM] latents shape", latents.shape)
        print("[DIM] mels shape", mels.shape)
        print("[DIM] eegs shape", eegs.shape)
        
        save_path = os.path.join(save_dir, f'data_batch_{i}.npz')
        save_data(eegs, mels, latents, save_path)


# Call this function before the training loop
train_dir = 'datasets/speech_encoding/training'
val_dir = 'datasets/speech_encoding/validation'
process_and_save_data(train_loader, train_dir)
print("Done processing and saving training data.")
process_and_save_data(val_loader, val_dir)
print("Done processing and saving validation data.")
