# Creates speech features: mel spectrogram (mel) and envelope (env)
import glob
import os
import numpy as np
import json
from task1_match_mismatch.util import envelope, mel_spectrogram
import task1_match_mismatch.util as util
# Directory to save the speech features
current_dir = os.getcwd()
save_cache_dir = os.path.join(current_dir, 'data_dir/train_dir/speech_features')
os.makedirs(save_cache_dir, exist_ok=True)

# Source speech directory containing raw speech
# The path to the challenge dataset is in /util/dataset_root_dir.json
os.chdir('..')
dataset_path_file = os.path.join(os.getcwd(), 'util', 'dataset_root_dir.json')
os.chdir('create_data')
with open(dataset_path_file, 'r') as f:
    dataset_root_dir = json.load(f)
source_speech_dir = os.path.join(dataset_root_dir, 'train', 'stimuli')
speech_files = glob.glob(os.path.join(source_speech_dir, '*.npz'))

for file in speech_files:
    # Loop over each speech file and create envelope and mel spectrogram
    # and save them
    stimulus = file.split('/')[-1].split('.')[0]
    save_path = os.path.join(save_cache_dir, stimulus + '.npz')

    # If the cache already exists then skip
    if not os.path.isfile(save_path):
        print(file)
        # Calculate envelope and mel spectrogram
        env = envelope.calculate_envelope(file)
        mel = mel_spectrogram.calculate_mel_spectrogram(file)
        # Save caches
        np.savez(save_path, env=env, mel=mel)



