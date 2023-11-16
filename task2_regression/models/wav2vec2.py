import numpy as np
from IPython.display import Audio
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Tokenizer, Wav2Vec2FeatureExtractor
import torchaudio
from transformers import Wav2Vec2Model

# data = np.load('stimuli/audiobook_1.npz')
# waveform = data['audio']


def SpeechEncoder(waveform):
    # Load the pre-trained Wav2Vec 2.0 model
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-large-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")

    # Resample waveform if needed (Wav2Vec2 is trained on 16 kHz)
    if waveform.shape[-1] != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=48000, new_freq=16000)
        waveform = resampler(waveform)

    # Process the waveform
    input_values = processor(
        waveform, sampling_rate=16000, return_tensors="pt").input_values

    # Get the latent representation
    with torch.no_grad():
        latent_representation = model(input_values).last_hidden_state

    return latent_representation


def SpeechFeatureExtractor(waveform):
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
