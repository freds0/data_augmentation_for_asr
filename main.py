import argparse
import tqdm
import os
import glob
import random
from utils.config import load_config
from data_augmentation import DataAugmentation
from audio_augmentation import AudioAugmentation
from spec_augmentation import SpecAugmentation

from datetime import datetime

random.seed(datetime.now())
sampling_rate = 16000

def insert_noise(dataset_dir, dataset_csv, output_folder, config):

    data_aug_options = {i: config.aug_data['aug_options'][i] for i in range(0, len(config.aug_data['aug_options']))}

    del config.aug_data["aug_options"]

    sa = SpecAugmentation(config.audio) 
    aa = AudioAugmentation(sampling_rate)
    da = DataAugmentation(sa, aa, sampling_rate, **config.aug_data)

    #metadata_csv = os.path.join(dataset_dir, dataset_csv)
    with open(dataset_csv, "r") as f:
        dataset_files = f.readlines()[1:]

    with open(config.aug_data['noises_filepath'], "r") as f:
        noise_file = f.readlines()

    os.makedirs(output_folder, exist_ok=True)
    output_file = open(os.path.join(output_folder, 'metadata.csv'), 'a')
    separator = '|'
    random.shuffle(dataset_files)

    for line in tqdm.tqdm(dataset_files):

        choice = random.choice([i for i in range(0, len(data_aug_options))])
        filebase, filesize, text2,  = line.split(separator)
        filepath = os.path.join(dataset_dir, filebase.replace('.wav', '') + '.wav') 
        filename = filepath.replace('/', '_').replace('.wav', '').replace('.', '') + '.wav'        

        if data_aug_options[choice] == 'noise':
            data_aug = da.insert_white_noise(filepath)
            filename = 'noise-{}'.format(filename)

        elif data_aug_options[choice] == 'shift':
            data_aug = da.shift_audio(filepath)
            filename = 'roll-{}'.format(filename)
 
        elif data_aug_options[choice] == 'stretch':
            data_aug = da.stretching_audio(filepath)
            filename = 'stretch-{}'.format(filename)

        elif data_aug_options[choice] == 'pitch':
            data_aug = da.changing_pitch(filepath)
            filename = 'pitch-{}'.format(filename)

        elif data_aug_options[choice] == 'external_noise':
            data_aug = da.insert_external_noise(filepath)
            filename = 'ex-{}'.format(filename)   

        elif data_aug_options[choice] == 'freq_mask':
            data_aug = da.insert_spectrogram_noise(filepath)
            filename = 'spec-{}'.format(filename)   

        elif data_aug_options[choice] == 'full_mix':
            data_aug = da.insert_mix_noise(filepath)
            filename = 'full-{}'.format(filename)   

        elif data_aug_options[choice] == 'lite_mix':
            data_aug = da.insert_lite_mix_noise(filepath)
            filename = 'lite_mix-{}'.format(filename)   

        output_wav_file = os.path.join(output_folder, 'wavs', filename)
        folder = os.path.dirname(output_wav_file)
        os.makedirs(folder, exist_ok=True)
        # Write augmentated sounds
        da.write_audio(output_wav_file, data_aug)
        
        #aa.plot_waveform(new_data, new_filepath.replace('.wav', '.png'), data)
        filesize = os.path.getsize(output_wav_file)
        line = separator.join([output_wav_file, str(filesize), text2.strip()])
        output_file.write(line + '\n')
             


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./') 
    parser.add_argument('-d', '--dataset_dir', default='', help='Name of csv file')   
    parser.add_argument('-i', '--input_csv', default='dataset-22k/metadata.csv', help='Name of csv file')   
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help="json file with configurations")
    parser.add_argument('-o', '--output_path', default='output', help='Name of output folder')   
    args = parser.parse_args()

    config = load_config(args.config_path)
    output_folder = os.path.join(args.base_dir, args.output_path)
    insert_noise(args.dataset_dir, args.input_csv, output_folder, config)


if __name__ == "__main__":
    main()
