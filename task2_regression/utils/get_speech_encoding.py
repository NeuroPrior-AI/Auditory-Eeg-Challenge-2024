import os
import numpy as np
from tqdm import tqdm  # tqdm is a library for progress bars
import sys
import time
sys.path.append('/home/naturaldx/auditory-eeg-challenge-2024-code')

from task2_regression.models.SpeechAutoEncoder import SpeechAutoEncoder
from task2_regression.configs.SAE import model_config, train_config
from util.data_loader import *
from util.wav2vec2 import mel_to_audio, speech_encoder, speech_encoder_pytorch


train_loader, val_loader = create_train_val_loader(batch_size=64)

def save_data(latents, file_path):
    """ Save the latents tensor to a file """
    # Detach the tensor and move it to CPU
    latents = latents.detach().cpu().numpy()
    np.savez(file_path, latents=latents)

def process_and_save_data(train_loader, save_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    """ Process and save the data """
    # total_batches = len(train_loader)
    for i, (eegs, mels) in enumerate(tqdm(train_loader, desc="Processing Batches", unit="batch")):
        eegs, mels = convert_to_torch(eegs, mels, device=device)

        # latents = []
        # total_mels = 64
        # for j, mel in enumerate(tqdm(mels, desc=f"Processing Mel Spectrograms in Batch {i}", unit="mel")):
        #     # Start timing for mel to audio conversion
        #     audio = mel_to_audio(mel)

        #     print("audio shape: ", audio.shape)
        #     audio = torch.tensor(audio)
        #     audio = audio.unsqueeze(0)

        #     latent = speech_encoder_pytorch(audio, sampling_rate=48000)
        #     latents.append(latent.squeeze(0))  # Remove the batch dimension

        #     # Log for each mel processing
        #     if j % 10 == 0 or j == total_mels - 1:  # Log every 10 mels or the last mel
        #         print(f"[Batch {i}, Mel {j}/{total_mels}] Processed mel spectrogram to latent")

        # latents = torch.stack(latents)  # Stack along a new batch dimension

        print("mels shape: ", mels.shape)
        # audio = mel_to_audio(mels)
        # audio = torch.tensor(audio)
        # latents = speech_encoder_pytorch(audio, sampling_rate=64)
        latents = torch.randn(64, 512)
        
        # Log dimensions of tensors
        print(f"[Batch {i}] latents shape: {latents.shape}, mels shape: {mels.shape}, eegs shape: {eegs.shape}")

        save_path = os.path.join(save_dir, f'data_batch_{i}.npz')
        save_data(latents, save_path)

        # Log saving status
        print(f"[LOG] Finished saving data batch {i}")
            
            
# def process_and_save_data(train_loader, save_dir):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     """ Process and save the data """
#     for i, (eegs, mels) in enumerate(train_loader):
#         eegs, mels = convert_to_torch(eegs, mels, device=device)

#         # print("[DIM] mels shape", mels.shape)
#         # waveforms = [mel_to_audio(mel) for mel in mels]
#         # waveforms = torch.tensor(waveforms)
#         # print("[DIM] waveforms shape", waveforms.shape)
#         # latents = speech_encoder_pytorch(waveforms, sampling_rate=48000)
#         # latents = speech_encoder(waveforms, sampling_rate=48000)
        
#         latents = []
#         for mel in mels:
#             audio = mel_to_audio(mel)
#             audio = torch.tensor(audio)
#             audio = audio.unsqueeze(0)
#             # print("[DIM] audio shape", audio.shape)
#             latent = speech_encoder_pytorch(audio, sampling_rate=48000)
#             # print("[DIM] latent shape", latent.shape)
#             latents.append(latent.squeeze(0))  # Remove the batch dimension

#         latents = torch.stack(latents)  # Stack along a new batch dimension
#         print("[DIM] latents shape", latents.shape)
#         print("[DIM] mels shape", mels.shape)
#         print("[DIM] eegs shape", eegs.shape)
        
#         save_path = os.path.join(save_dir, f'data_batch_{i}.npz')
#         save_data(eegs, mels, latents, save_path)
        
#         print("[LOG] Finished saving data batch", i)


# Call this function before the training loop
train_dir = 'datasets/speech_encoding/temp'
val_dir = 'datasets/speech_encoding/validation'
process_and_save_data(train_loader, train_dir)
print("Done processing and saving training data.")
process_and_save_data(val_loader, val_dir)
print("Done processing and saving validation data.")
