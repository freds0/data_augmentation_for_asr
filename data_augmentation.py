import argparse
import os
import numpy as np
import random
from librosa.core import load
from utils.config import load_config
from spec_augmentation import SpecAugmentation
from audio_augmentation import AudioAugmentation
from scipy.io.wavfile import write
from datetime import datetime

random.seed(datetime.now())
sampling_rate = 16000

class DataAugmentation():

    def __init__(self,  sa: SpecAugmentation,
                        aa: AudioAugmentation,
                        sr,
                        aug_options,
                        noises_filepath,
                        noise_range_min,
                        noise_range_max,
                        pitch_range_min,
                        pitch_range_max,
                        stretch_range_min,
                        stretch_range_max,
                        external_noise_range_min,
                        external_noise_range_max,
                        shift_roll_range_min,
                        shift_roll_range_max,
                        num_freq_mask,
                        freq_masking_min_percentage,
                        freq_masking_max_percentage,
                        num_time_mask,
                        time_masking_min_percentage,
                        time_masking_max_percentage
    ):
        self.sa = sa
        self.aa = aa
        self.sr = sr
        self.noise_range_min = noise_range_min
        self.noise_range_max = noise_range_max
        self.pitch_range_min = pitch_range_min
        self.pitch_range_max = pitch_range_max
        self.stretch_range_min = stretch_range_min
        self.stretch_range_max = stretch_range_max
        self.external_noise_range_min = external_noise_range_min
        self.external_noise_range_max = external_noise_range_max
        self.shift_roll_range_min = shift_roll_range_min
        self.shift_roll_range_max = shift_roll_range_max
        self.num_freq_mask = num_freq_mask
        self.freq_masking_min_percentage = freq_masking_min_percentage
        self.freq_masking_max_percentage = freq_masking_max_percentage
        self.num_time_mask = num_time_mask
        self.time_masking_min_percentage = time_masking_min_percentage
        self.time_masking_max_percentage = time_masking_max_percentage
        self.noises_list = self.get_noises_list(noises_filepath)

    def get_noises_list(self, noise_filepath):
        f = open(noise_filepath, "r")
        noises_list = f.readlines()
        f.close()        
        return noises_list

    def read_audio(self, filepath):
        data, _ = load(filepath, self.sr)
        return data

    def write_audio(self, filepath, data):
        data = (data * 32767).astype(np.int16) # Convert to int16
        write(filepath, self.sr, data)

    def insert_white_noise(self, filepath):
        data = self.read_audio(filepath)
        # Adding white noise to sound
        noise_rate = random.uniform(self.noise_range_min, self.noise_range_max)
        data_aug = self.aa.add_noise(data, noise_rate)
        return data_aug

    def shift_audio(self, filepath):
        data = self.read_audio(filepath)
        # Shifting the sound
        shift_rate = random.uniform(self.shift_roll_range_min, self.shift_roll_range_max)
        data_aug = self.aa.shift(data, shift_rate)
        # Adding white noise to sound
        noise_rate = random.uniform(self.noise_range_min, self.noise_range_max)
        data_aug = self.aa.add_noise(data_aug, noise_rate / 2)    
        return data_aug

    def stretching_audio(self, filepath):
        data = self.read_audio(filepath)
        # Stretching the sound
        stretch_rate = random.uniform(self.stretch_range_min, self.stretch_range_max)
        data_aug = self.aa.stretch(data, stretch_rate)
        # Adding white noise to sound
        noise_rate = random.uniform(self.noise_range_min, self.noise_range_max)
        data_aug = self.aa.add_noise(data_aug, noise_rate / 10)
        return data_aug

    def changing_pitch(self, filepath):
        data = self.read_audio(filepath)
        # Changing pitch
        pitch_rate = random.uniform(self.pitch_range_min, self.pitch_range_max)
        data_aug = self.aa.pitch(data, pitch_rate)
        # Adding noise to sound
        noise_rate = random.uniform(self.noise_range_min, self.noise_range_max)
        data_aug = self.aa.add_noise(data_aug, noise_rate / 10)
        return data_aug

    def insert_external_noise(self, filepath):
        data = self.read_audio(filepath)
        # inserting asr-noises
        noise_filepath = random.choice(self.noises_list).strip()
        noise = self.aa.read_audio_file(noise_filepath)
        ex_noise_rate = random.uniform(self.external_noise_range_min, self.external_noise_range_max)
        data_aug = self.aa.add_external_noise(data, noise, ex_noise_rate)
        # Adding noise to sound
        noise_rate = random.uniform(self.noise_range_min, self.noise_range_max)
        data_aug = self.aa.add_noise(data_aug, noise_rate / 10)  
        return data_aug

    def insert_spectrogram_noise(self, filepath):
        # Read wavefile
        spec, phase = self.sa.get_spectrogram_phase(filepath)

        # Apply data augmentation on spectogram domain
        freq_percentage = random.uniform(self.freq_masking_min_percentage, self.freq_masking_max_percentage)
        spec_aug = self.sa.freq_spec_augment(spec, self.num_freq_mask, freq_percentage)

        # Apply data augmentation on time domain
        time_percentage = random.uniform(self.time_masking_min_percentage, self.time_masking_max_percentage)   
        spec_aug = self.sa.time_spec_augment(spec_aug, self.num_time_mask, time_percentage)
        # Get wav
        data_aug = self.sa.inv_spectrogram(spec_aug, phase)
        return data_aug


    def insert_mix_noise(self, filepath):
        # Read wavefile
        spec, phase = self.sa.get_spectrogram_phase(filepath)

        # Apply data augmentation on spectogram domain
        freq_percentage = random.uniform(self.freq_masking_min_percentage, self.freq_masking_max_percentage)
        spec_aug = self.sa.freq_spec_augment(spec, self.num_freq_mask, freq_percentage)

        # Apply data augmentation on time domain
        time_percentage = random.uniform(self.time_masking_min_percentage, self.time_masking_max_percentage)   
        spec_aug = self.sa.time_spec_augment(spec_aug, self.num_time_mask, time_percentage)
        # Get wav
        data_aug = self.sa.inv_spectrogram(spec_aug, phase)

        # Changing pitch
        pitch_rate = random.uniform(self.pitch_range_min, self.pitch_range_max)
        data_aug = self.aa.pitch(data_aug, pitch_rate)

        # Stretching the sound
        stretch_rate = random.uniform(self.stretch_range_min, self.stretch_range_max)
        data_aug = self.aa.stretch(data_aug, stretch_rate)

        # Shifting the sound
        shift_rate = random.uniform(self.shift_roll_range_min, self.shift_roll_range_max)
        data_aug = self.aa.shift(data_aug, shift_rate)

        # Adding noise to sound
        noise_rate = random.uniform(self.noise_range_min, self.noise_range_max)
        data_aug = self.aa.add_noise(data_aug, noise_rate)  

        # inserting asr-noises
        noise_filepath = random.choice(self.noises_list).strip()
        noise = self.aa.read_audio_file(noise_filepath)
        ex_noise_rate = random.uniform(self.external_noise_range_min, self.external_noise_range_max)
        data_aug = self.aa.add_external_noise(data_aug, noise, ex_noise_rate)

        return data_aug


    def insert_lite_mix_noise(self, filepath):
        # Read wavefile
        spec, phase = self.sa.get_spectrogram_phase(filepath)

        # Apply data augmentation on spectogram domain
        freq_percentage = random.uniform(self.freq_masking_min_percentage, self.freq_masking_max_percentage)
        spec_aug = self.sa.freq_spec_augment(spec, self.num_freq_mask, freq_percentage)

        # Apply data augmentation on time domain
        time_percentage = random.uniform(self.time_masking_min_percentage, self.time_masking_max_percentage)   
        spec_aug = self.sa.time_spec_augment(spec_aug, self.num_time_mask, time_percentage)
        # Get wav
        data_aug = self.sa.inv_spectrogram(spec_aug, phase)

        # Adding noise to sound
        noise_rate = random.uniform(self.noise_range_min, self.noise_range_max) / 10
        data_aug = self.aa.add_noise(data_aug, noise_rate)  

        # inserting asr-noises
        noise_filepath = random.choice(self.noises_list).strip()
        noise = self.aa.read_audio_file(noise_filepath)
        ex_noise_rate = random.uniform(self.external_noise_range_min, self.external_noise_range_max) / 10
        data_aug = self.aa.add_external_noise(data_aug, noise, ex_noise_rate)

        return data_aug

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./') 
    parser.add_argument('-i', '--input_file', type=str, required=True, help="input wav file")
    parser.add_argument('-o', '--output_file', type=str, required=True, help="output wav data augmented file")
    parser.add_argument('-c', '--config_path', type=str, required=True, help="json file with configurations")
    args = parser.parse_args()

    config = load_config(args.config_path)  

    sa = SpecAugmentation(config.audio) 
    aa = AudioAugmentation(sampling_rate)
    da = DataAugmentation(sa, aa, sampling_rate, **config.data_aug)

    filepath = args.input_file
    data_aug = da.insert_external_noise(filepath)
    da.write_audio(args.output_file, data_aug)

if __name__ == "__main__":
    main()