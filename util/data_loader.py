import os
import glob
import json
import torch        # Paul: if torch is imported after tensorflow then I get SegmentationFault when trying to put a pytorch tensor to GPU, I don't know why...
import tensorflow as tf
from dataset_generator import DataGenerator, create_tf_dataset

# Disable TensorFlow warnings
tf.get_logger().setLevel('ERROR')


def convert_to_torch(data, label, device='cpu'):
    """
    Convert TensorFlow EagerTensors to PyTorch tensors and optionally load them on a specified device.

    Parameters:
    data: The data tensor from TensorFlow.
    label: The label tensor from TensorFlow.
    device: The PyTorch device ('cuda' or 'cpu') to load the tensors onto.

    Returns:
    A tuple of PyTorch tensors (data, label) possibly loaded on the specified device.
    """
    # Convert the TensorFlow EagerTensors to numpy arrays
    data = data.numpy()
    label = label.numpy()
    # Convert numpy arrays to PyTorch tensors
    data = torch.from_numpy(data).to(device)
    label = torch.from_numpy(label).to(device)  
    return data, label
    

def create_test_loader(window_length=3840, hop_length=1920):
    # Get the path to the config gile
    util_folder = os.path.dirname(__file__)
    config_path = os.path.join(util_folder, 'config.json')    

    # Load the config
    with open(config_path) as fp:
        config = json.load(fp)

    # Provide the path of the dataset
    # which is split already to train, val, test
    data_folder = os.path.join(config["dataset_folder"],config["derivatives_folder"],  config["split_folder"])
    stimulus_features = ["mel"]
    features = ["eeg"] + stimulus_features
    
    # Create a dataset generator for each test subject
    test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    # Get all different subjects from the test set
    subjects = list(set([os.path.basename(x).split("_-_")[1] for x in test_files]))
    datasets_test = {}
    # Create a generator for each subject
    for sub in subjects:
        files_test_sub = [f for f in test_files if sub in os.path.basename(f)]
        tf_test_generator = DataGenerator(files_test_sub, window_length)
        tf_dataset = create_tf_dataset(tf_test_generator, window_length, None, hop_length, batch_size=1, data_types=(tf.float32, tf.float32), feature_dims=(64, 10))
        datasets_test[sub] = tf_dataset

    return datasets_test


def create_train_val_loader(window_length=3840, hop_length=1920, batch_size=64):
    # Get the path to the config gile
    util_folder = os.path.dirname(__file__)
    config_path = os.path.join(util_folder, 'config.json')    

    # Load the config
    with open(config_path) as fp:
        config = json.load(fp)

    # Provide the path of the dataset
    # which is split already to train, val, test
    data_folder = os.path.join(config["dataset_folder"],config["derivatives_folder"],  config["split_folder"])
    stimulus_features = ["mel"]
    features = ["eeg"] + stimulus_features

    train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features ]
    train_generator = DataGenerator(train_files, window_length)
    dataset_train = create_tf_dataset(train_generator, window_length, None, hop_length, batch_size, data_types=(tf.float32, tf.float32), feature_dims=(64, 10))

    val_files = [x for x in glob.glob(os.path.join(data_folder, "val_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    val_generator = DataGenerator(val_files, window_length)
    dataset_val = create_tf_dataset(val_generator, window_length, None, hop_length, batch_size, data_types=(tf.float32, tf.float32), feature_dims=(64, 10))

    return dataset_train, dataset_val


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}.")

    # Note: these datasets contain Tensorflow tensors, so convert_to_torch() needs to be called to convert them to PyTorch tensors
    train_loader, val_loader = create_train_val_loader(batch_size=64)
    test_loader = create_test_loader()

    # Train loop:
    print(f"------------- Train set overview -------------")
    for i, (eegs, mels) in enumerate(train_loader):

        eegs, mels = convert_to_torch(eegs, mels, device=device)
        print(f"{i}: eegs: {eegs.shape}, mels: {mels.shape}")
        print("eegs tensor on GPU:", eegs.is_cuda)
        print("mels tensor on GPU:", mels.is_cuda)

        # Rest of training code
        # ...
        break


    # Test set:
    print(f"------------- Test set overview -------------")
    for subject, data in test_loader.items():
        print(f"Subject {subject}:")
        data = [x for x in data]
        eegs, mels = tf.concat([ x[0] for x in data], axis=0), tf.concat([ x[1] for x in data], axis=0)
        eegs, mels = convert_to_torch(eegs, mels, device=device)
        print(f"eegs: {eegs.shape}, labels: {mels.shape}")