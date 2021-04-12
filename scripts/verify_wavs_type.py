import argparse
import soundfile as sf
from glob import glob
from tqdm import tqdm
import scipy.io.wavfile as wav
import numpy as np
from os import makedirs
from os.path import dirname, exists

def verify(args):
    for wavfile in tqdm(sorted(glob(args.base_dir + '/**/**/**/*.wav'))):
        ob = sf.SoundFile(wavfile)
        if ob.subtype != 'PCM_16' or ob.channels != 1 or ob.samplerate != 16000:
            print(wavfile)
            '''
            sr, data = wav.read(wavfile)
            filepath = wavfile.replace('wavs', 'new_wavs')
            mydir = dirname(filepath)
            if not exists(mydir):
                makedirs(mydir)
            data = data*32768
            wav.write(filepath, sr, data.astype(np.int16))           
            '''
            print('Sample rate: {}'.format(ob.samplerate))
            print('Channels: {}'.format(ob.channels))
            print('Subtype: {}'.format(ob.subtype))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--csv_file', default='metadata.csv', help='Name of csv file')
    args = parser.parse_args()
    verify(args)

if __name__ == "__main__":
    main()
