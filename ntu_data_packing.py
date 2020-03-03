import argparse 
import os.path as osp
import numpy as np
import pickle as pkl
from glob import glob

def main():
    all_file_list = glob(osp.join(args.ntu_vibe_dir, f'*/*.pkl'))
    print(all_file_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ntu_vibe_dir', type=str, 
                        help='Path to the directory with the VIBE extraction of NTU videos')
    args = parser.parse_args()

    main(args)