import argparse
import glob
import os
from os import makedirs
from os.path import join, exists, dirname
import tqdm

number_bits = 16
encoding = "signed-integer"
number_channels = 1

def convert(filepath, force, sr, basename):
        folder = dirname(filepath)
        filename = filepath.split('/')[-1]
        new_folder = folder.replace(basename, basename + '-' + sr)
        new_filepath = join(new_folder, filename)
        if not exists(new_folder) and (force):
            makedirs(new_folder)
        if force:
            os.system("sox %s -V0 -c %d -r %d -b %d -e %s %s"% (filepath, int(number_channels), int(sr), number_bits, encoding, new_filepath))
        else:
            print("sox %s -V0 -c %d -r %d -b %d -e %s %s"% (filepath, int(number_channels), int(sr), number_bits, encoding, new_filepath))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--sr', default='22050')
    parser.add_argument('--force', action='store_true', default=False)

    args = parser.parse_args()

    for filepath in tqdm.tqdm(sorted(glob.glob(args.base_dir + '/**/*.wav'))):
        convert(filepath, args.force, args.sr, args.base_dir)

if __name__ == "__main__":
    main()
