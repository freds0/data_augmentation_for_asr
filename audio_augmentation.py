import librosa
import argparse
from utils.config import load_config
from scipy.io.wavfile import read, write
import numpy as np
import matplotlib.pyplot as plt
import random
# Reference: https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6
# Reference: https://www.kaggle.com/CVxTz/audio-data-augmentation
# Reference: https://github.com/alibugra/audio-data-augmentation
from datetime import datetime
MAX_WAV_VALUE=32767

random.seed(datetime.now())
sampling_rate = 16000

class AudioAugmentation:
    def __init__(self, sr):
        self.sr = sr

    def read_audio_file(self, file_path):
        data = librosa.core.load(file_path, self.sr)[0]
        #_, data = read(file_path)
        input_length = len(data)
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data

    def write_audio_file(self, filename, data):
        #librosa.output.write_wav(filename, data, self.sr)
        data = data*MAX_WAV_VALUE
        write(filename, self.sr, data.astype(np.int16))

    # Inserting white noise
    def add_noise(self, data, rate=0.01):
        #noise = np.random.randn(len(data))
        noise = np.random.normal(0,1,len(data))
        augmented_data = data + rate * noise
        return augmented_data

    # Interting external wav file noise
    def add_external_noise(self, data, noise, rate):
        if len(noise) < len(data):
            #noise = np.pad(noise, (0, max(0, len(data) - len(noise))), "constant")
            while(len(noise) < len(data)):
                noise =  np.concatenate((noise, noise[:len(data)-len(noise)]), axis=None)
        else:
            noise = noise[:len(data)]

        augmented_data = data + rate * noise
        return augmented_data

    # Shifting Time
    def shift(self, data, shift_rate, shift_direction = 'both'):
        # The idea of shifting time is very simple. It just shift audio to left/right with a random second. If shifting audio to left (fast forward) with x seconds, first x seconds will mark as 0 (i.e. silence). If shifting audio to right (back forward) with x seconds, last x seconds will mark as 0 (i.e. silence).
        #rate_roll = int(self.sr/shift_max)
        #return np.roll(data, rate_roll)
        #shift = np.random.randint(self.sr * shift_max)
        shift = int(self.sr * shift_rate)
        if shift_direction == 'right':
            shift = -shift
        elif shift_direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1: # shift_direction = 'right' if direction == 1 else 'left' 
                shift = -shift    

        augmented_data = np.roll(data, shift)
        # Set to silence for heading/ tailing
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        return augmented_data

    # Changing Speed
    def stretch(self, data, rate=1):
        input_length = len(data)
        # Speed of speech
        augmented_data = librosa.effects.time_stretch(data, rate)
        if len(augmented_data) > input_length:
            augmented_data = augmented_data[:input_length]
        else:
            augmented_data = np.pad(augmented_data, (0, max(0, input_length - len(augmented_data))), "constant")
        return augmented_data

    # Changing Pitch
    def pitch(self, data, pitch_factor):
        augmented_data = librosa.effects.pitch_shift(data, self.sr, pitch_factor)
        return augmented_data
                                
    def plot_waveform(self, aug_waveform, path, clean_waveform=None, title=None, split_title=False, max_len=None):
        # Reference: https://github.com/Rayhane-mamah/Tacotron-2/blob/master/tacotron/utils/plot.py
	    if max_len is not None:
            aug_waveform = aug_waveform[:max_len]
            clean_waveform = clean_waveform[:max_len]

        if split_title:
            title = split_title_line(title)

        fig = plt.figure(figsize=(10, 8))
        # Set common labels
        fig.text(0.5, 0.18, title, horizontalalignment='center', fontsize=16)

        #target spectrogram subplot
	    if clean_waveform is not None:
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            im = ax1.plot(np.linspace(0, 1, len(clean_waveform)), clean_waveform)
            plt.ylabel('Amplitude')

            ax1.set_title('Clean Waveform')
            ax2.set_title('Augmented Waveform')

            ax1.set_ylim([-1, 1])
            ax2.set_ylim([-1, 1])

        else:
            ax2 = fig.add_subplot(211)
            ax2.set_ylim([-1, 1])

        im = ax2.plot(np.linspace(0, 1, len(aug_waveform)), aug_waveform)
        plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.savefig(path, format='png')
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./') 
    parser.add_argument('-i', '--input_file', type=str, required=True, help="input wav file")
    parser.add_argument('-o', '--output_dir', type=str, required=True, help="output dir of data augmented files")
    parser.add_argument('-c', '--config_path', type=str, required=True, help="json file with configurations")
    args = parser.parse_args()
    # Read config
    config = load_config(args.config_path)  

    # Read noise files list
    with open(config.data_aug['noises_filepath'], "r") as f:
        noises_file_content = f.readlines()

    # Generate AudioAugmentation object
    aa = AudioAugmentation(sampling_rate)
    data = aa.read_audio_file(args.input_file)

    from os.path import basename, dirname, join
    from os import makedirs

    output_file = basename(args.input_file).split('.')[0]
    # Adding white noise to sound
    filename = join(args.output_dir, output_file + '_noise.wav')
    makedirs(dirname(filename), exist_ok=True)
    noise_rate = random.uniform(config.data_aug['noise_range_min'], config.data_aug['noise_range_max'])
    data_aug = aa.add_noise(data, noise_rate)
    aa.write_audio_file(filename, data_aug)
    aa.plot_waveform(data_aug, filename.replace('.wav', '.png'), data)

    # Shifting the sound
    filename = join(args.output_dir, output_file + '_shift.wav')
    shift_rate = random.uniform(config.data_aug['shift_roll_range_min'], config.data_aug['shift_roll_range_max'])
    data_aug = aa.shift(data, shift_rate)
    aa.write_audio_file(filename, data_aug)
    aa.plot_waveform(data_aug, filename.replace('.wav', '.png'), data)

    # Stretching the sound
    filename = join(args.output_dir, output_file + '_stretch.wav')
    stretch_rate = random.uniform(config.data_aug['stretch_range_min'], config.data_aug['stretch_range_max'])
    data_aug = aa.stretch(data, stretch_rate)
    aa.write_audio_file(filename, data_aug)
    aa.plot_waveform(data_aug, filename.replace('.wav', '.png'), data)

    # Changing pitch
    filename = join(args.output_dir, output_file + '_pitch.wav')
    pitch_rate = random.uniform(config.data_aug['pitch_range_min'], config.data_aug['pitch_range_max'])
    data_aug = aa.pitch(data, pitch_rate)
    aa.write_audio_file(filename, data_aug)
    aa.plot_waveform(data_aug, filename.replace('.wav', '.png'), data)

    # inserting external noises
    filename = join(args.output_dir, output_file + '_exnoise.wav')
    noise_filepath = random.choice(noises_file_content).strip()
    noise = aa.read_audio_file(noise_filepath)
    ex_noise_rate = random.uniform(config.data_aug['external_noise_range_min'], config.data_aug['external_noise_range_max'])
    aa.plot_waveform(noise*ex_noise_rate, filename.replace('.wav', '_noise.png'), data)
    data_aug = aa.add_external_noise(data, noise, ex_noise_rate)
    aa.write_audio_file(filename, data_aug)
    aa.plot_waveform(data_aug, filename.replace('.wav', '.png'), data)

if __name__ == "__main__":
    main()