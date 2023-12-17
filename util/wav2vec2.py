import numpy as np
from IPython.display import Audio
import soundfile as sf
import torch
import torchaudio
from torchaudio.utils import download_asset
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Tokenizer, Wav2Vec2FeatureExtractor
import torchaudio
from transformers import Wav2Vec2Model

import librosa
import math

def mel_to_audio(mel, sr=48000, n_fft=2048, hop_length=750, win_length=1200, n_iter=32, n_mels=10, length=None):
    """
    Convert mel spectrogram to waveform using GPU acceleration.
    Parameters are the same as in the original function.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Ensure mel is a PyTorch tensor
    if not isinstance(mel, torch.Tensor):
        mel = torch.tensor(mel)

    # Move mel to GPU if available
    mel = mel.to(device)
    
    # Transpose mel spectrogram if necessary
    if mel.shape[0] != n_mels:
        mel = mel.transpose(0, 1)
        
    # Initialize the InverseMelScale transform
    n_stft = int((n_fft//2) + 1)
    inverse_melscale_transform = torchaudio.transforms.InverseMelScale(n_stft=n_stft, n_mels=n_mels, sample_rate=sr).to(device)

    # Apply the transform to convert mel spectrogram to linear frequency spectrogram
    linear_spec = inverse_melscale_transform(mel)
    
    # Use Griffin-Lim algorithm for phase reconstruction
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop_length, win_length=win_length).to(device)
    audio = griffin_lim(linear_spec)
    
    # Truncate or pad the waveform to the desired length
    if length is not None:
        if len(audio) > length:
            audio = audio[:length]
        elif len(audio) < length:
            audio = torch.nn.functional.pad(audio, (0, length - len(audio)))
        
    # Move audio back to CPU for numpy compatibility if needed
    audio = audio.cpu()

    return audio.numpy()


def speech_encoder(waveform, sampling_rate=16000):
    """Convert the speech waveform to latent representation using Wav2Vec2
    pretrained model.

    Parameters
    ----------
    waveform : torch.Tensor
        The waveform tensor. Can be a single sample or a batch of samples.
    sampling_rate : int, optional
        The sampling rate of the waveform, by default 16000

    Returns
    -------
    latent_representation : torch.Tensor
        The latent representation from the Wav2Vec2 model.
    """
    
    # Ensure waveform is a torch.Tensor
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform)

    # Load the pre-trained Wav2Vec 2.0 model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")

    # print("[Y] waveform shape before re-sampling", waveform.shape)
    # Resample waveform if needed
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        waveform = resampler(waveform)

    # print("[Y] waveform shape before processing 1", waveform.shape)
    
    # Process the waveform
    input_values = processor(waveform, sampling_rate=16000, return_tensors="pt").input_values
    
    if input_values.ndim > 2:
        input_values = input_values.squeeze(0)
         
    # print("[Y] input_values shape after processing", input_values.shape)
    # Get the latent representation
    with torch.no_grad():
        latent_representation = model(input_values).last_hidden_state

    return latent_representation


def speech_feature_extractor(waveform):
    """ IGNORE THIS FUNCTION
    """
    # Load the model and feature extractor
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53")

    # Resample waveform if needed (Wav2Vec2 is trained on 16 kHz)
    if waveform.shape[-1] != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=48000, new_freq=16000)
        waveform = resampler(waveform)

    # Process the waveform
    input_values = feature_extractor(
        waveform, return_tensors="pt", padding="longest", sampling_rate=16000, do_normalize=True).input_values

    # Get the latent representation
    with torch.no_grad():
        latent_representation = model(input_values).last_hidden_state

    return latent_representation


def speech_encoder_pytorch(waveform, sampling_rate=48000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
    model = bundle.get_model().to(device)

    waveform = waveform.to(device)
    if sampling_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, sampling_rate, bundle.sample_rate)

    with torch.inference_mode():
        features, _ = model.extract_features(waveform)

    # return features[-1]
    # Take the average of the last 4 layers
    # Convert the list of the last four layers into a tensor
    last_four_layers = torch.stack(features[-4:])

    # Compute the mean across these layers
    # The dimension for mean should be 0 since you stacked the layers along a new dimension at the front
    mean_features = torch.mean(last_four_layers, dim=0)

    # Apply a convolution to mean_features for dimensionality reduction
    # The convolution should have 512 output channels
    # The input shape is (batch_size, sequence_length, 1024) and the output shape should be (batch_size, sequence_length, 512)
    
    # Create the convolutional layer
    conv = torch.nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1)

    # Move the convolutional layer to the same device as your input tensor
    conv = conv.to(mean_features.device)

    # Apply the convolutional layer to the input tensor
    mean_features = conv(mean_features.transpose(1, 2))
    mean_features = mean_features.transpose(1, 2)

    return mean_features