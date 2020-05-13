import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import glob
import numpy as np
from multiprocessing import Pool
import pickle as pk
import uuid
import shutil
import argparse

def extract_amass(gpu_idx, jobs, output_dir):
    # os.system(job)
    pk_name = "/tmp/{}.pkl".format(uuid.uuid1())
    pk.dump(jobs, open(pk_name, "wb"))
    cmd = "CUDA_VISIBLE_DEVICES={}  python  process_directory.py  --video_dir  {} --output_folder {}".format(gpu_idx, pk_name, output_dir)
    os.system(cmd)
    
    os.remove(pk_name)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_dir', type=str,
                        help='input directory full of videos')

    parser.add_argument('--output_dir', type=str,
                        help='output folder to write results')
                        
    args = parser.parse_args()

    print("Running As list")
    gpu_idxs = [0, 1, 2, 3]
    num_jobs = 8

    video_base = args.video_dir
    output_dir = args.output_dir

    paths = glob.glob(os.path.join(video_base, "*.mp4"))
    jobs = []
    for video_dir in sorted(paths):
        video_output = os.path.join(output_dir, video_dir.split("/")[-1][:-4] + ".pkl")
        if osp.isfile(video_output):
            print(video_output, "output exists, next...")
        else:
            jobs.append(video_dir)
        # print("************************ ",video_dir)


    chunk = np.ceil(len(jobs)/num_jobs).astype(int)
    jobs= [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
    args = [(gpu_idxs[ i % (len(gpu_idxs))], jobs[i], output_dir) for i in range(len(jobs))]

    try:
        pool = Pool(num_jobs)   # multi-processing
        pool.starmap(extract_amass, args)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()