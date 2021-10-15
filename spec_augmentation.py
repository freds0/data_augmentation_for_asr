# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Performs data augmentation on spectrogram audio.
#
# References:
#
# Paper SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition (available at https://arxiv.org/abs/1904.08779).
# https://www.kaggle.com/davids1992/specaugment-quick-implementation/
#
# (C) 2021 Frederico Oliveira, UFMT/UFG
# Released under GNU Public License (GPL)
# email fred.santos.oliveira@gmail.com
#
import argparse
import random
import librosa
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from utils.config import load_config
from datetime import datetime

# Setting seed.
random.seed(datetime.now())

# Constants
MAX_WAV_VALUE=32767

class SpecAugmentation:
    '''
    Class that performs data augmentation on audio spectrogram.
    '''
    def __init__(self, config):
        '''
        Initializes an object of the class.
        Args:
            config:
        '''

        self.segment_length      = config['segment_length']
        self.sr                  = config['sample_rate']
        self.num_freq            = config['num_freq']
        self.hop_length          = config['hop_length']
        self.win_length          = config['win_length']

    def get_spectrogram_phase(self, filepath):
        '''
        Load a spectrogram and phase from audio file using Librosa.
        Args:
            filepath: path of the file to be loaded.

        Returns:
            spec_mag: magnitude spectrogram.
            phase: the phase of audio.
        '''

        y, sr = librosa.core.load(filepath)
        spec =  librosa.stft(y, n_fft=self.num_freq, hop_length=self.hop_length, win_length=self.win_length)
        spec_mag, phase = librosa.magphase(spec)
        return spec_mag, phase

    def write_audio_file(self, spec, phase, filepath):
        '''
        Saves an audio file given the spectrogram and the phase extracted initially.
        Args:
            spec: magnitude spectrogram.
            phase: original phase of audio.
            filepath: filpath to save the audio file.

        '''
        enhanced = librosa.istft(spec * phase, hop_length=self.hop_length, win_length=self.win_length)
        '''
        # Using old version of scipy to save the file.
        wav_norm = wav * (MAX_WAV_VALUE / max(0.01, np.max(np.abs(enhanced))))
        scipy.io.wavfile.write(path, self.sampling_rate, wav_norm.astype(np.int16))
        '''
        enhanced = enhanced * MAX_WAV_VALUE # normalizes [0..1] the audio file.
        wavfile.write(filepath, self.sr, enhanced.astype(np.int16))

    def freq_spec_augment(self, spec: np.ndarray, num_freq_mask, freq_percentage):
        '''
        Data augmentation performing masking on the frequency axis. Reference: https://www.kaggle.com/davids1992/specaugment-quick-implementation/
        Args:
            spec: magnitude spectrogram.
            num_freq_mask: total amount of masking to be performed on the frequency axis.
            freq_percentage: percentage of each frequency axis masking.

        Returns:
            spec: magnitude spectrogram with frequency masking data augmentation.
        '''
        spec = spec.copy()
        for i in range(num_freq_mask):
            # get the number of frames and the number of frequencies.
            all_frames_num, all_freqs_num = spec.shape
            # defines the amount of masking given a percentage.
            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            # defines which frequency will be masked.
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            # rounds to an integer.
            f0 = int(f0)
            # masks the frequency by assigning zero.
            spec[:, f0:f0 + num_freqs_to_mask] = 0
        return spec

    def time_spec_augment(self, spec: np.ndarray, num_time_mask, time_percentage):
        '''
        Data augmentation performing masking on the time axis. Reference: Reference: https://www.kaggle.com/davids1992/specaugment-quick-implementation/
        Args:
            spec: magnitude spectrogram.
            num_time_mask: total amount of masking to be performed on the time axis.
            time_percentage: percentage of each time axis masking.

        Returns:
            spec: magnitude spectrogram with time masking data augmentation.
        '''
        spec = spec.copy()
        for i in range(num_time_mask):
            # get the number of frames and the number of frequencies.
            all_frames_num, all_freqs_num = spec.shape
            # defines the amount of masking given a percentage.
            num_frames_to_mask = int(time_percentage * all_frames_num)
            # defines which instant of time will be masked.
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            # rounds to an integer.
            t0 = int(t0)
            # masks the instant of time by assigning zero.
            spec[t0:t0 + num_frames_to_mask, :] = 0   

        return spec

    def plot_spectrogram(self, aug_spectrogram, path, clean_spectrogram=None, title=None, split_title=False, max_len=None, auto_aspect=True):
        '''
        Plot a waveform compared to a clean waveform. Reference: https://github.com/Rayhane-mamah/Tacotron-2/blob/master/tacotron/utils/plot.py
        Args:
            aug_spectrogram: audio data spectrogram augmented.
            path: path to save figure.
            clean_waveform:  clean audio data spectrogram.
            title: title of the image.
            split_title: True or False.
            max_len: maximum length of the audio data to be plotted.
            auto_aspect: True or False.
        '''
        if max_len is not None:
            aug_spectrogram = aug_spectrogram[:max_len]
            clean_spectrogram = clean_spectrogram[:max_len]

        if split_title:
            title = split_title_line(title)

        fig = plt.figure(figsize=(10, 8))
        # Set common labels
        fig.text(0.5, 0.18, title, horizontalalignment='center', fontsize=16)

        #target spectrogram subplot
        if clean_spectrogram is not None:
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)

            if auto_aspect:
                im = ax1.imshow(np.rot90(clean_spectrogram), aspect='auto', interpolation='none')
            else:
                im = ax1.imshow(np.rot90(clean_spectrogram), interpolation='none')
            ax1.set_title('Clean Spectrogram')
            fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
            ax2.set_title('Augmented Spectrogram')
        else:
            ax2 = fig.add_subplot(211)

        if auto_aspect:
            im = ax2.imshow(np.rot90(aug_spectrogram), aspect='auto', interpolation='none')
        else:
            im = ax2.imshow(np.rot90(aug_spectrogram), interpolation='none')
        fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax2)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(path, format='png')
        plt.close()

    def inv_spectrogram(self, spec_aug, phase):
        '''
        Return an audio file given the spectrogram and the phase extracted initially.
        Args:
            spec_aug: augmented magnitude spectrogram.
            phase: original phase of audio.

        '''
        enhanced = librosa.istft(spec_aug * phase, hop_length=self.hop_length, win_length=self.win_length)
        return enhanced


def main():
    '''
    Example of using this class.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./') 
    parser.add_argument('-i', '--input_file', type=str, required=True, help="input wav file")
    parser.add_argument('-o', '--output_file', type=str, required=True, help="output wav data augmented file")
    parser.add_argument('-c', '--config_path', type=str, required=True, help="json file with configurations")
    args = parser.parse_args()
    # Read config
    c = load_config(args.config_path)

    # Generate SpecAugmentation object
    sa = SpecAugmentation(c.spectrogram)

    # Read wav file
    spec, phase = sa.get_spectrogram_phase(args.input_file)

    # Apply data augmentation on spectogram domain
    freq_percentage = random.uniform(c.data_aug['freq_masking_min_percentage'], c.data_aug['freq_masking_max_percentage'])
    spec_aug = sa.freq_spec_augment(spec, c.data_aug['num_freq_mask'], freq_percentage)
    # Plot    
    sa.plot_spectrogram(aug_spectrogram=spec_aug, path=args.output_file.replace('.wav', '1.png'), clean_spectrogram=spec, title='Spec Domain', split_title=False)
    # Saving to wav file
    sa.write_audio_file(spec_aug, phase, args.output_file.replace('.wav', '1.wav'))

    # Apply data augmentation on time domain
    time_percentage = random.uniform(c.data_aug['time_masking_min_percentage'], c.data_aug['time_masking_max_percentage'])   
    spec_aug = sa.time_spec_augment(spec, c.data_aug['num_time_mask'], time_percentage)
    # Plot
    sa.plot_spectrogram(aug_spectrogram=spec_aug, path=args.output_file.replace('.wav', '2.png'), clean_spectrogram=spec, title='Time Domain', split_title=False)
    # Saving to wav file
    sa.write_audio_file(spec_aug, phase, args.output_file.replace('.wav', '2.wav'))

    # Apply both data augmentation on time and frequency domain
    freq_percentage = random.uniform(c.data_aug['freq_masking_min_percentage'], c.data_aug['freq_masking_max_percentage'])
    spec_aug = sa.freq_spec_augment(spec, c.data_aug['num_freq_mask'], freq_percentage)

    time_percentage = random.uniform(c.data_aug['time_masking_min_percentage'], c.data_aug['time_masking_max_percentage'])   
    spec_aug = sa.time_spec_augment(spec_aug, c.data_aug['num_time_mask'], time_percentage)
    # Plot
    sa.plot_spectrogram(aug_spectrogram=spec_aug, path=args.output_file.replace('.wav', '3.png'), clean_spectrogram=spec, title='Time and Frequency Domain', split_title=False)

    # Saving to wav file
    sa.write_audio_file(spec, phase, args.output_file.replace('.wav', '3.wav'))

if __name__ == "__main__":
    main()