import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('/home/naturaldx/auditory-eeg-challenge-2024-code')

from task2_regression.models.SpeechAutoEncoder import SpeechAutoEncoder
from task2_regression.configs.SAE import model_config, train_config
from util.data_loader import *
from util.wav2vec2 import mel_to_audio, speech_encoder

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}.")

    # Initialize the model
    model = SpeechAutoEncoder(h=model_config['decoder_config'], latent_dim=model_config['latent_dim']).to(device)
    optimizer = model_config['optimizer'](model.parameters(), lr=model_config['lr'][1])
    loss_fn = model_config['loss_fn']

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=train_config['log_path'])

    # Assuming train_loader, val_loader, and test_loader are already defined
    train_loader, val_loader = create_train_val_loader(batch_size=64)
    
    # Training loop
    for epoch in range(train_config['max_epochs']):
        model.train()
        for i, (eegs, mels) in enumerate(train_loader):
            # mels = mels.to(device)
            # Convert the data and labels to PyTorch tensors and load them on the device (GPU or CPU)
            eegs, mels = convert_to_torch(eegs, mels, device=device)

            print("[X] mels shape", mels.shape)
            # for each data insdie the batch, encode mel spectrogram to latent representation
            # initialize a list to store latent representations
            waveforms = []
            for j in range(len(mels)):
                # print("[X] mels shape", mels[j].shape)
                waveform = mel_to_audio(mels[j])
                # print("[X] waveform shape", waveform.shape)
                waveforms.append(waveform)
            latents = speech_encoder(waveforms, sampling_rate=model_config['sampling_rate'])
            
            # Transpose latents to (batch_size, latent_dim, length)
            latents = latents.transpose(1, 2)
            mels = mels.transpose(1, 2)
            print("[X] latents shape", latents.shape)
            
            # Move latent representation to GPU
            latents = latents.to(device)
            
            # Forward pass
            # outputs = model(mels)
            outputs = model(latents)
            loss = loss_fn(outputs, mels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log training loss
            writer.add_scalar('Training loss', loss.item(), epoch * 64 + i)

        # Validation loop
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for mels, _ in val_loader:
                mels = mels.to(device)

                # Forward pass
                outputs = model(mels)
                loss = loss_fn(outputs, mels)
                total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            writer.add_scalar('Validation loss', avg_val_loss, epoch)

        print(f'Epoch [{epoch+1}/{train_config["max_epochs"]}], Train Loss: {loss.item()}, Val Loss: {avg_val_loss}')

    # Close the TensorBoard writer
    writer.close()

    # Save the trained model
    torch.save(model.state_dict(), f"{train_config['log_path']}/final_model.pth")

def test(model, test_loader, device):
    model.eval()
    total_test_loss = 0
    loss_fn = model_config['loss_fn']
    test_loader = create_test_loader()
    
    # Convert the data and labels to PyTorch tensors and load them on the device (GPU or CPU)
    test_loader = convert_to_torch(test_loader, device)
    
    with torch.no_grad():
        for subject, data in test_loader.items():
            print(f"Subject {subject}:")
            data = [x for x in data]
            # If data is NoneType, report it and skip to next subject
            
            eegs, mels = tf.concat([ x[0] for x in data], axis=0), tf.concat([ x[1] for x in data], axis=0)
            eegs, mels = convert_to_torch(eegs, mels, device=device)
            
            # Encode mel spectrogram to latent representation
            waveform = mel_to_audio(mels)
            latent = speech_encoder(waveform, sampling_rate=model_config['sampling_rate'])
            
            # Move latent representation to GPU
            latent = latent.to(device)
            
            # Forward pass
            # outputs = model(mels)
            outputs = model(latent)
            loss = loss_fn(outputs, mels)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f'Average Test Loss: {avg_test_loss}')

if __name__ == "__main__":
    train()
