import numpy as np
import matplotlib.pyplot as plt


class Visualizer():

    # plot_specific allows you to plot specific channels of the eeg (select channels from 1-64 inclusive)
    def plot_eeg(file_path, out_path, plot_specific=[]):
        eeg_data = np.load(file_path)
        print(f"eeg_data.shape: {eeg_data.shape}")

        if len(plot_specific) != 0:
            # Print some statistics for each selected channel
            print(f"Per channel statistics:")
            for i in plot_specific:
                print(f"channel {i}: mean: {np.mean(eeg_data[i - 1, :])}, sd: {np.std(eeg_data[i - 1, :])}, min: {np.min(eeg_data[i - 1, :])}, max: {np.max(eeg_data[i - 1, :])}")
            
            # Plot specified channels
            plt.figure(figsize=(100, len(plot_specific) * 2))  
            for idx, i in enumerate(plot_specific):
                plt.subplot(len(plot_specific), 1, idx+1)
                plt.plot(eeg_data[i - 1], linewidth=0.3)
                plt.title(f'Channel {i}')
                plt.grid(True)
                plt.tight_layout()
        else:
            # Print some statistics for every channel
            print(f"Per channel statistics:")
            for i in range(64):
                print(f"channel {i + 1}: mean: {np.mean(eeg_data[i, :])}, sd: {np.std(eeg_data[i, :])}, min: {np.min(eeg_data[i, :])}, max: {np.max(eeg_data[i, :])}")
            
            # Plot every channel
            plt.figure(figsize=(100, 64 * 2))  
            for i in range(64):
                plt.subplot(64, 1, i+1)
                plt.plot(eeg_data[i], linewidth=0.1)
                plt.title(f'Channel {i+1}')
                plt.grid(True)
                plt.tight_layout()
    
        plt.savefig(out_path, dpi=300)
        plt.show()


    def plot_mel(file_path, out_path):
        mel_data = np.load(file_path)
        print(f"mel_data.shape: {mel_data.shape}")

        # Print some statistics for every channel
        print(f"Per channel statistics:")
        for i in range(10):
            print(f"channel {i + 1}: mean: {np.mean(mel_data[:, i])}, sd: {np.std(mel_data[:, i])}, min: {np.min(mel_data[:, i])}, max: {np.max(mel_data[:, i])}")

        # Plotting
        plt.figure(figsize=(100, 5))  # Adjust figure size as needed
        plt.imshow(mel_data.T, aspect='auto', origin='lower', cmap='jet')
        plt.colorbar(label='Magnitude')
        plt.ylabel('Mel Frequency Bands')
        plt.xlabel('Time Frames')
        plt.title('Mel Spectrogram')
        plt.tight_layout()

        plt.savefig(out_path, dpi=300)  # You can adjust dpi for resolution
        plt.show()


    def plot_envelope(file_path, out_path):
        envelope_data = np.load(file_path)
        print(f"envelope_data.shape: {envelope_data.shape}")

        plt.figure(figsize=(100, 6))
        plt.plot(envelope_data, linewidth=0.5)  # Adjust linewidth for line thickness
        plt.title('Envelope Visualization')
        plt.xlabel('Time Frames')
        plt.ylabel('Envelope Value')
        plt.grid(True)  # Add gridlines
        plt.tight_layout()

        plt.savefig(out_path, dpi=300)  # You can adjust dpi for resolution
        plt.show()


if __name__ == "__main__":
    file_path = "/h/335/paulslss300/audeeg_data/derivatives/preprocessed_eeg/sub-001/ses-shortstories01/sub-001_ses-shortstories01_task-listeningActive_run-01_desc-preproc-audio-audiobook_5_1_eeg.npy"
    out_path = "/h/335/paulslss300/auditory-eeg-challenge-2024-code/plots/eeg_plot.png"
    Visualizer.plot_eeg(file_path, out_path, [1])

    file_path = "/h/335/paulslss300/audeeg_data/derivatives/preprocessed_stimuli/audiobook_1_-_mel.npy"
    out_path = "/h/335/paulslss300/auditory-eeg-challenge-2024-code/plots/mel_plot.png"
    Visualizer.plot_mel(file_path, out_path)

    file_path = "/h/335/paulslss300/audeeg_data/derivatives/preprocessed_stimuli/audiobook_1_-_envelope.npy"    
    out_path = "/h/335/paulslss300/auditory-eeg-challenge-2024-code/plots/envelope_plot.png"
    Visualizer.plot_envelope(file_path, out_path)

