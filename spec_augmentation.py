import argparse
import os
from scipy.io import wavfile
import librosa
import numpy as np
import matplotlib.pyplot as plt
import random
#from utils.audio_processor import WrapperAudioProcessor as AudioProcessor 
from utils.config import load_config
import torch
from datetime import datetime

MAX_WAV_VALUE=32767
random.seed(datetime.now())

class SpecAugmentation:

    def __init__(self, config):
        self.segment_length      = config['segment_length']
        self.sr                  = config['sample_rate']
        self.filter_length       = config['filter_length']
        self.num_freq            = config['num_freq']
        self.hop_length          = config['hop_length']
        self.win_length          = config['win_length']

    def get_spectrogram_phase(self, filepath):
        #wav = self.ap.load_wav(filepath)
        y, sr = librosa.core.load(filepath)
        spec =  librosa.stft(y, n_fft=self.num_freq, hop_length=self.hop_length, win_length=self.win_length)
        spec_mag, phase = librosa.magphase(spec)
        #spectrogram, phase = self.ap.get_spec_from_audio(wav, return_phase=True)
        return spec_mag, phase

    def write_audio_file(self, spec, phase, filepath):
        #double_spectrogram = torch.from_numpy(np.array([spec, spec]))
        #double_phase = torch.from_numpy(np.array([phase, phase]))
        #wav = self.ap.torch_inv_spectrogram(double_spectrogram, double_phase)
        #wav = wav.cpu().detach().numpy()
        #self.ap.save_wav(wav[0], filepath) 
        enhanced = librosa.istft(spec * phase, hop_length=self.hop_length, win_length=self.win_length)
        #wav_norm = wav * (MAX_WAV_VALUE / max(0.01, np.max(np.abs(enhanced))))
        #scipy.io.wavfile.write(path, self.sampling_rate, wav_norm.astype(np.int16))
        enhanced = enhanced*MAX_WAV_VALUE
        wavfile.write(filepath, self.sr, enhanced.astype(np.int16))

    def freq_spec_augment(self, spec: np.ndarray, num_freq_mask, freq_percentage):
        # Reference: https://www.kaggle.com/davids1992/specaugment-quick-implementation/
        spec = spec.copy()
        for i in range(num_freq_mask):
            all_frames_num, all_freqs_num = spec.shape            
            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = 0
        return spec

    def time_spec_augment(self, spec: np.ndarray, num_time_mask, time_percentage):
        # Reference: https://www.kaggle.com/davids1992/specaugment-quick-implementation/
        spec = spec.copy()
        for i in range(num_time_mask):
            all_frames_num, all_freqs_num = spec.shape         
            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = 0   

        return spec

    def plot_spectrogram(self, aug_spectrogram, path, clean_spectrogram=None, title=None, split_title=False, max_len=None, auto_aspect=True):
        # Reference: https://github.com/Rayhane-mamah/Tacotron-2/blob/master/tacotron/utils/plot.py
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./') 
    parser.add_argument('-i', '--input_file', type=str, required=True, help="input wav file")
    parser.add_argument('-o', '--output_file', type=str, required=True, help="output wav data augmented file")
    parser.add_argument('-c', '--config_path', type=str, required=True, help="json file with configurations")
    args = parser.parse_args()
    args = parser.parse_args()
    # Read config
    c = load_config(args.config_path)  

    # Generate SpecAugmentation object
    #sa = SpecAugmentation(c.audio)
    sa = SpecAugmentation(c.spectrogram)

    # Read wav file
    spec, phase = sa.get_spectrogram_phase(args.input_file)
    '''
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
    '''
    # Saving to wav file
    sa.write_audio_file(spec, phase, args.output_file.replace('.wav', '3.wav'))

if __name__ == "__main__":
    main()