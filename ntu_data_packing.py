import argparse 
import os.path as osp
import numpy as np
import joblib
import pickle as pkl
from tqdm import tqdm
from glob import glob

TRAINING_SUBJECT_IDS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
NTU_NUM_CLASSES = 60

def get_subject_id(filename):
    return int(filename[9:12])


def get_class_id(filename):
    return int(filename[17:20]) - 1


def build_one_hot(num_classes, class_idx):
    one_hot = np.zeros(num_classes)
    one_hot[class_idx] = 1
    return one_hot


def main(args):
    # List down all the pickles
    all_file_list = glob(osp.join(args.ntu_vibe_dir, f'*/*.pkl'))

    training_data = dict()
    testing_data = dict()
    target_fields = ['pose', 'betas']

    pbar = tqdm(all_file_list)

    for file_path in pbar:
        filename, _ = osp.splitext(osp.basename(file_path))
        subject_id = get_subject_id(filename)
        pbar.set_description(f'Processing {filename}')

        pkl_data = joblib.load(file_path)
        data = pkl_data[list(pkl_data.keys())[0]]

        extracted_data = dict()
        for field in target_fields:
            extracted_data[field] = data[field]
        extracted_data['label'] = get_class_id(filename)
        extracted_data['label_onehot'] = build_one_hot(NTU_NUM_CLASSES, extracted_data['label'])

        if subject_id in TRAINING_SUBJECT_IDS:
            training_data[filename] = extracted_data
        else:
            testing_data[filename] = extracted_data

        break    
    pbar.close()

    pkl.dump(training_data, open('train_vibe_ntu.pkl', 'wb'))
    pkl.dump(testing_data, open('test_vibe_ntu.pkl', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ntu_vibe_dir', type=str, 
                        help='Path to the directory with the VIBE extraction of NTU videos')
    parser.add_argument('--output_dir', type=str,
                        help='Path to output directory where the train/test pkl files will be stored')
                        
    args = parser.parse_args()

    main(args)