from pathlib import Path
from torch import optim
import torch
from transformers import Data2VecAudioConfig,AutoConfig

import sys
sys.path.append('/home/naturaldx/auditory-eeg-challenge-2024-code')
from task2_regression.models.SpeechAutoEncoder import SpeechAutoEncoder



MODEL_NAME = 'SAE'
loss = torch.nn.SmoothL1Loss()
light_model = SpeechAutoEncoder
sampling_rate = 16000
base_path = Path(__file__).parent.parent

# Define your hyperparameters object (or dictionary) here
# h = {
#     'resblock_kernel_sizes': [3, 7, 11],  # Example values
#     'resblock_dilation_sizes': [(1, 3), (3, 5), (5, 7)],  # Example values
#     'resblock': '1',  # Example value
#     'upsample_rates': [2, 2, 2, 2, 2, 2, 5],
#     'upsample_kernel_sizes': [2, 2, 3, 3, 3, 3, 10],
#     'upsample_initial_channel': 768,
#     # Add other necessary hyperparameters here
# }
latent_dim = 1024  # Example value
decoder_config = AutoConfig.from_pretrained(base_path/"cache"/"models"/"hifigan",local_files_only=True, trust_remote_code=True,        
        upsample_rates=[2, 2, 2, 2, 2, 2, 2, 2],
        upsample_initial_channel=1024,
        upsample_kernel_sizes=[2, 2, 2, 2, 2, 2, 2, 2],
        model_in_dim=1024,
        sampling_rate=sampling_rate)
model_config = dict(
    decoder_config=decoder_config,
    # h=h,
    latent_dim=latent_dim,
    optimizer=optim.AdamW,
    loss_fn=loss,
    lr=[0.0000001,0.001],
    sampling_rate = sampling_rate
)

# Keep optimizer and other training related configurations separate
train_config = dict(
    gpus=1,
    mode = 'train',
    log_path = './experiments/'+MODEL_NAME,
    max_epochs= 200,
    # train_batch_size= 2,
    # eval_batch_size = 1,
    deterministic=True,
)
# nohup python train.py --config config.segment_NIKENet_2D --mode continue --gpu 0 --checkpoint experiment/segment_NIKENetSeg_2D/last.ckpt > seg.out &
# nohup python train.py --config config.segment_NIKENet_2D --mode train --gpu 1&
# train ['887', '388', '429', '438', '125', '403', '674', '211', '432', '576', '732', '789', '723', '910', '891', '556', '801', '673', '379', '470']
# val ['481', '206', '551', '196', '176', '530']