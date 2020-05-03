import argparse 
import os.path as osp
import numpy as np
import joblib
import pickle as pkl
from tqdm import tqdm
from glob import glob

MUTUAL_ACTIONS_IDS = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
TRAINING_SUBJECT_IDS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
NTU_NUM_CLASSES = 60
TRAIN_OUTPUT_FILENAME = 'train_vibe_ntu.pkl'
TEST_OUTPUT_FILENAME = 'test_vibe_ntu.pkl'
EMPTY_OUTPUT_FILENAME = 'empty_vibe_ntu.lst'
ERROR_OUTPUT_FILENAME = 'error_vibe_ntu.lst'


def get_subject_id(filename):
    p_idx = filename.find('P') + 1
    return int(filename[p_idx:p_idx+3])


def get_class_id(filename):
    a_idx = filename.find('A') + 1
    return int(filename[a_idx:a_idx+3]) - 1


def build_one_hot(num_classes, class_idx):
    one_hot = np.zeros(num_classes)
    one_hot[class_idx] = 1
    return one_hot


def pad_pose_seq(min_len, pose_seq):
    seq_len, pose_len = pose_seq.shape
    if seq_len < min_len:
        padding_len = min_len - seq_len
        padding = np.repeat(pose_seq[-1,:].reshape((72,1)), padding_len, axis=1).swapaxes(0,1)
        return np.vstack((pose_seq, padding))
    return pose_seq

def pad_trans_seq(min_len, trans_seq):
    seq_len, trans_len = trans_seq.shape
    if seq_len < min_len:
        padding_len = min_len - seq_len
        padding = np.repeat(trans_seq[-1,:].reshape((trans_len,1)), padding_len, axis=1).swapaxes(0,1)
        return np.vstack((trans_seq, padding))
    return trans_seq


def main(args):
    # List down all the pickles
    all_file_list = glob(osp.join(args.ntu_vibe_dir, f'*/*.pkl'))

    training_data = dict()
    testing_data = dict()
    empty_file_list = list()
    error_file_list = list()

    pbar = tqdm(all_file_list)
    sample_counter = dict(train=0, test=0, empty=0)

    for file_idx, file_path in enumerate(pbar):
        filename, _ = osp.splitext(osp.basename(file_path))
        try:
            subject_id = get_subject_id(filename)
            class_id = get_class_id(filename)
        except ValueError:
            print(f'Could not extract info from: {filename}')
            error_file_list.append(file_path)
            continue

        pbar.set_description(f'Processing {filename}  train:{sample_counter["train"]}  test:{sample_counter["test"]}  empty:{sample_counter["empty"]}')

        pkl_data = joblib.load(file_path)
        if len(pkl_data.keys()) < 1:
            empty_file_list.append(file_path)
            continue

        data = pkl_data[list(pkl_data.keys())[0]]

        extracted_data = dict()
        extracted_data['pose'] = pad_pose_seq(args.min_seq_len, data['pose'])
        extracted_data['betas'] = data['betas']
        extracted_data['pred_cam'] = pad_trans_seq(args.min_seq_len, data['pred_cam'])
        extracted_data['orig_cam'] = pad_trans_seq(args.min_seq_len, data['orig_cam'])
        extracted_data['label'] = class_id
        extracted_data['label_onehot'] = build_one_hot(NTU_NUM_CLASSES, extracted_data['label'])

        if subject_id in TRAINING_SUBJECT_IDS:
            training_data[filename] = extracted_data
        else:
            testing_data[filename] = extracted_data 

        sample_counter['train'] = len(training_data.keys())
        sample_counter['test'] = len(testing_data.keys())
        sample_counter['empty'] = len(empty_file_list)

        if file_idx > 0 and file_idx % 1000 == 0:
            pkl.dump(training_data, open(osp.join(args.output_dir, TRAIN_OUTPUT_FILENAME), 'wb'))
            pkl.dump(testing_data, open(osp.join(args.output_dir, TEST_OUTPUT_FILENAME), 'wb'))
            with open(osp.join(args.output_dir, EMPTY_OUTPUT_FILENAME), 'w') as empty_file:
                empty_file.write('\n'.join(empty_file_list))
            with open(osp.join(args.output_dir, ERROR_OUTPUT_FILENAME), 'w') as error_file:
                error_file.write('\n'.join(error_file_list))

    pbar.close()

    pkl.dump(training_data, open(osp.join(args.output_dir, TRAIN_OUTPUT_FILENAME), 'wb'))
    pkl.dump(testing_data, open(osp.join(args.output_dir, TEST_OUTPUT_FILENAME), 'wb'))
    with open(osp.join(args.output_dir, EMPTY_OUTPUT_FILENAME), 'w') as empty_file:
        empty_file.write('\n'.join(empty_file_list))
    with open(osp.join(args.output_dir, ERROR_OUTPUT_FILENAME), 'w') as error_file:
        error_file.write('\n'.join(error_file_list))

    print(f'Training samples:  {len(training_data.keys())}')
    print(f'Testing samples:   {len(testing_data.keys())}')
    print(f'Empty files:       {len(empty_file_list)}')
    print(f'Error files:       {len(error_file_list)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ntu_vibe_dir', type=str, 
                        help='Path to the directory with the VIBE extraction of NTU videos')
    parser.add_argument('--min_seq_len', type=int, default=90,
                        help='Minimum number of frames requires in pose sequence, if lower the sequence will be padded')
    parser.add_argument('--output_dir', type=str,
                        help='Path to output directory where the train/test pkl files will be stored')
                        
    args = parser.parse_args()

    main(args)