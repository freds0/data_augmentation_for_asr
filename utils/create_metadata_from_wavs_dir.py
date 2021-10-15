import argparse
from glob import glob
import os
import shutil
from os.path import join, getsize

def execute(args):
    output_file = open(join(args.base_dir, args.output_file), 'w')
    separator = '|'
    
    for i, filepath in enumerate(sorted(glob(join(args.base_dir, args.input_dir) + '/*.wav'))):
        text = ''
        size = getsize(filepath)
        line = separator.join([filepath,str(size), text])
        output_file.write(line + '\n')

    output_file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--input_dir', default='./input')
    parser.add_argument('--output_file', default='./metadata.csv')
    args = parser.parse_args()
    execute(args)

if __name__ == "__main__":
    main()
