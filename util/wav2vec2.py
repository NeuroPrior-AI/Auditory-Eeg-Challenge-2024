import numpy as np
from IPython.display import Audio
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Tokenizer, Wav2Vec2FeatureExtractor
import torchaudio
from transformers import Wav2Vec2Model

import librosa


def mel_to_audio(mel, sr=16000, n_fft=512, hop_length=160, win_length=400, n_iter=32, length=None):
    """Convert mel spectrogram to waveform using inverse.mel_to_audio from librosa
    Parameters
    ----------
    mel : np.ndarray
        Mel spectrogram
    sr : int
        Sampling rate
    n_fft : int
        FFT window size
    hop_length : int
        Hop length
    win_length : int
        Window length
    n_iter : int
        Number of iterations
    length : int
        Length of the output waveform
    Returns
    -------
    audio : np.ndarray
        Waveform
    """
    # Check if mel is a PyTorch tensor and on GPU, then move to CPU
    if isinstance(mel, torch.Tensor):
        if mel.is_cuda:
            mel = mel.cpu()
        mel = mel.numpy()
    elif not isinstance(mel, np.ndarray):
        mel = np.array(mel)

    # Convert mel spectrogram to waveform
    # audio = librosa.feature.inverse.mel_to_audio(
    #     mel, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_iter=n_iter)
    audio = librosa.feature.inverse.mel_to_audio(mel)

    # Truncate the waveform to the given length
    if length is not None and len(audio) > length:
        audio = audio[:length]
        
    return audio


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
