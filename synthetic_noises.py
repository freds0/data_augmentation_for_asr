# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Class that performs synthetic noise creation for data augmentation.
# source: https://github.com/makermovement/3.5-Sensor2Phone/blob/master/generate_any_audio.py
#
#
# (C) 2021 Frederico Oliveira, UFMT/UFG
# Released under GNU Public License (GPL)
# email fred.santos.oliveira@gmail.com
#
#
import argparse
import numpy as np
from scipy import signal as sg
from scipy.io.wavfile import write
import librosa
from os.path import join
import random
from datetime import datetime
from utils.config import load_config
# Setting seed.
random.seed(datetime.now())
# Constants
sampling_rate = 16000
MAX_WAV_VALUE=32767

class SyntheticNoise:
    '''
    Class that performs synthetic noise creation for data augmentation.
    '''

    def __init__(self, config):
        '''
        Initializes an object of the class.
        Args:
            config:
        '''
        self.sr = config['sample_rate']

    def read_audio_file(self, filepath):
        '''
        Loads a clean audio speech file which will be carried out data augmentation.
        Args:
            filepath: path of the file to be loaded.

        Returns:
            data: numpy audio data time series.

        '''
        data = librosa.core.load(filepath, self.sr)[0]
        return data

    def sine_wave(self, frequency = 2205, length = 10):
        '''
        Generate a sine wave.
        Args:
            frequency:
            length:

        Returns:

        '''
        x = np.arange(length)
        y = 100 * np.sin(2 * np.pi * frequency * x / self.sr)
        return y

    def square_wave(self, frequency = 2205, length = 10):
        '''
        Generate a Square Wave.
        Args:
            frequency:
            length:

        Returns:

        '''
        x = np.arange(length)
        y = 100 * sg.square(2 * np.pi * frequency * x / self.sr )
        return y

    def duty_square_wave(self, frequency = 2205, length = 10):
        '''
        Generate a Square wave with Duty Cycle.
        Args:
            frequency:
            length:

        Returns:

        '''
        x = np.arange(length)
        y = 100* sg.square(2 * np.pi * frequency * x / self.sr , duty = 0.8)
        return y

    def sawtooth_wave(self, frequency = 2205, length = 10):
        '''
        Generate a Sawtooth wave.
        Args:
            frequency:
            length:

        Returns:

        '''
        x = np.arange(length)
        y = 100 * sg.sawtooth(2 * np.pi * frequency * x / self.sr )
        return y

    def write_audio_file(self, filename, data):
        '''
        Saves the audio data to a file.
        Args:
            filename: path to save the file
            data: audio data.
        '''

        data = data * MAX_WAV_VALUE
        write(filename, self.sr, data.astype(np.int16))
        # Librosa version
        # librosa.output.write_wav(filename, data, self.sr)

    def add_noise(self, data, noise_type, rate = 0.0001):
        '''
        Inserting white noise at audio data.
        Args:
            data: audio data.
            noise_type: The options are sine, square, duty_square or sawtooth.
            rate: noise intensity rate to be inserted.

        Returns:
            augmented_data: audio data with white noise inserted.
        '''
        f = random.randrange(10, self.sr / 10)
        l = len(data)
        if noise_type == 'sine':
            noise_data = self.sine_wave(frequency=f, length=l)
        elif noise_type == 'square':
            noise_data = self.square_wave(frequency=f, length=l)
        elif noise_type == 'duty_square':
            noise_data = self.duty_square_wave(frequency=f, length=l)
        elif noise_type == 'sawtooth':
            noise_data = self.sawtooth_wave(frequency=f, length=l)

        augmented_data = data + rate * noise_data
        return augmented_data

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
    config = load_config(args.config_path)

    sn = SyntheticNoise(config.audio)
    data = sn.read_audio_file(args.input_file)

    data_augmented = sn.add_noise(data, noise_type = 'sine', rate = 0.0001)
    sn.write_audio_file(join(args.base_dir, args.output_file + '_sine.wav'), data_augmented)

    data_augmented = sn.add_noise(data, noise_type = 'square', rate = 0.0001)
    sn.write_audio_file(join(args.base_dir,  args.output_file + '_square.wav'), data_augmented)

    data_augmented = sn.add_noise(data, noise_type = 'duty_square', rate = 0.0001)
    sn.write_audio_file(join(args.base_dir,  args.output_file + '_duty_square.wav'), data_augmented)

    data_augmented = sn.add_noise(data, noise_type = 'sawtooth', rate = 0.0001)
    sn.write_audio_file(join(args.base_dir, args.output_file +  '_sawtooth.wav'), data_augmented)

if __name__ == "__main__":
    main()