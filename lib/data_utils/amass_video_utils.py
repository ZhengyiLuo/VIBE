# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import os
import cv2
import torch
import joblib
import argparse
import numpy as np
import pickle as pkl
import os.path as osp
from tqdm import tqdm
import pickle as pk

from lib.models import spin
from lib.data_utils.kp_utils import *
from lib.core.config import VIBE_DB_DIR, VIBE_DATA_DIR
from lib.utils.smooth_bbox import get_smooth_bbox_params
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14
from lib.data_utils.feature_extractor import extract_features
from lib.utils.geometry import batch_rodrigues, rotation_matrix_to_angle_axis

from zen_renderer.utils.transform_utils import convert_orth_6d_to_aa
from zen_renderer.renderer.smpl_renderer import SMPL_Renderer

NUM_JOINTS = 24
VIS_THRESH = 0.3
MIN_KP = 6

def amass_to_dataset(k, v, smpl_renderer, set, device = ( torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu"))):
    img_dir = "/hdd/zen/data/ActBound/AMASS/Rendering/take3/images/"
    curr_vid_name = k.split("_")[0]
    vid_dir = osp.join(img_dir, curr_vid_name)
    if osp.isdir(vid_dir) and len(os.listdir(vid_dir)) == 150:
        curr_pose = v['pose']
        curr_pose_aa = convert_orth_6d_to_aa(torch.tensor(curr_pose).float().to(device))
        num_frames = curr_pose.shape[0]
        vid_name = np.repeat(np.array(k), num_frames)
        verts_perp, joints_prep, verts, joints, joint_36m = smpl_renderer.look_at_verts(curr_pose_aa)
        frame_id = np.linspace(0, num_frames - 1, num_frames).astype(int)

        j2d = joints_prep[:,H36M_TO_J14,:].detach().cpu().numpy()
        if set == "train":
            j3d = joints.detach().cpu().numpy()
        else:
            j3d = joint_36m[:,H36M_TO_J14,:].detach().cpu().numpy()
        
        bbox = None
        img_name = np.array([osp.join(vid_dir, "frame{:06d}.jpg".format(i)) for i in range(num_frames)])

        shape = np.zeros((num_frames, 10))
        pose = curr_pose_aa.detach().cpu().numpy()
        features = None
        valid = np.ones((num_frames, 10))
        return vid_name, frame_id, j3d, j2d, shape, pose, bbox, img_name, features, valid
    else:
        print("Empty Videos", vid_dir)
        return None, None, None, None, None, None, None, None, None, None,

        

def read_data(amass_data, set, debug=False, max_samples = -1):

    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'shape': [],
        'pose': [],
        'bbox': [],
        'img_name': [],
        'features': [],
        'valid': [],
    }
    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = spin.get_pretrained_hmr()
    smpl_renderer = SMPL_Renderer(device = device, image_size = 400, camera_mode = "look_at")
    for i, (k,v) in tqdm(enumerate(amass_data)):
        vid_name, frame_id, j3d, j2d, shape, pose, bbox, img_name, features, valid = amass_to_dataset(k, v, set = set, smpl_renderer = smpl_renderer)

        if not vid_name is None:
            bbox_params, time_pt1, time_pt2 = get_smooth_bbox_params(j2d, vis_thresh=VIS_THRESH, sigma=8)

            c_x = bbox_params[:,0]
            c_y = bbox_params[:,1]
            scale = bbox_params[:,2]
            w = h = 150. / scale
            w = h = h * 1.1
            bbox = np.vstack([c_x,c_y,w,h]).T
            # print('campose', campose_valid[time_pt1:time_pt2].shape)

            img_paths_array = img_name
            dataset['vid_name'].append(vid_name)
            dataset['frame_id'].append(frame_id)
            dataset['img_name'].append(img_name)
            dataset['joints3D'].append(j3d)
            dataset['joints2D'].append(j2d)
            dataset['shape'].append(shape)
            dataset['pose'].append(pose)
            dataset['bbox'].append(bbox)
            dataset['valid'].append(valid)

            features = extract_features(model, img_paths_array, bbox,
                                        kp_2d=j2d[time_pt1:time_pt2], debug=debug, dataset='3dpw', scale=1.2)
            dataset['features'].append(features)
                
        if max_samples != -1 and i > max_samples:
            break
    for k in dataset.keys():
        dataset[k] = np.concatenate(dataset[k])
        print(k, dataset[k].shape)

    # Filter out keypoints
    indices_to_use = np.where((dataset['joints2D'][:, :, 2] > VIS_THRESH).sum(-1) > MIN_KP)[0]
    for k in dataset.keys():
        dataset[k] = dataset[k][indices_to_use]

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/3dpw')
    args = parser.parse_args()

    debug = False

    amass_data_dir = '/hdd/zen/data/ActBound/AMASS/amass_take3.pkl'
    amass_data = pk.load(open(amass_data_dir, "rb"))

    np.random.seed(0)
    amass_items = np.array(list(amass_data.items()))
    all_data_idx = list(range(len(amass_items)))
    np.random.shuffle(all_data_idx)
    train_list = amass_items[all_data_idx[:int(len(all_data_idx) * 0.6)]]
    test_list = amass_items[all_data_idx[int(len(all_data_idx) * 0.6):int(len(all_data_idx) * 0.8)]]
    val_list = amass_items[all_data_idx[int(len(all_data_idx) * 0.8):]]
    

    # dataset = read_data(train_list, 'train')
    # joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'amass_rend_take3_train_db.pt'))

    dataset = read_data(val_list, 'val', max_samples=500)
    joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'amass_rend_take3_val_db.pt'))

    dataset = read_data(test_list, 'test', max_samples=500)
    joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'amass_rend_take3_test_db.pt'))

    
