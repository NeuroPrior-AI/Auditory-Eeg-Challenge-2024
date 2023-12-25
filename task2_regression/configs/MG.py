from pathlib import Path
import sys
import torch
from torch import optim
from transformers import AutoConfig

# Adding path to the system path
sys.path.append('/home/naturaldx/auditory-eeg-challenge-2024-code')

# Importing the model
from task2_regression.models.MelGenerator import EEGtoMelAutoEncoder

# Model and training configuration
MODEL_NAME = 'SAE'
sampling_rate = 64
base_path = Path(__file__).parent.parent

# Decoder configuration
decoder_config = AutoConfig.from_pretrained(
    base_path / "cache" / "models" / "hifigan",
    local_files_only=True,
    trust_remote_code=True,
    upsample_rates=[2, 2],
    upsample_initial_channel=64,
    upsample_kernel_sizes=[2, 2],
    model_in_dim=64,
    sampling_rate=sampling_rate
)

# Model configuration
model_config = {
    'decoder_config': decoder_config,
    'latent_dim': 64,
    'optimizer': optim.AdamW,
    'loss_fn': torch.nn.SmoothL1Loss(),
    'lr': [0.0000001, 0.001],
    'sampling_rate': sampling_rate
}

# Training configuration
train_config = {
    'gpus': 1,
    'mode': 'train',
    'log_path': './experiments/' + MODEL_NAME,
    'max_epochs': 200,
    'deterministic': True
}

# Ensure to use these configurations appropriately in your training loop.
