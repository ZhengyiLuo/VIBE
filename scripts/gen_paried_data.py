import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch
import joblib
import numpy as np
from tqdm import tqdm 
import h5py

from lib.dataset import *
from lib.models import VIBE
from lib.models.smpl import SMPL, SMPL_MODEL_DIR
from lib.core.config import parse_args
from torch.utils.data import DataLoader
from lib.core.config import VIBE_DATA_DIR, VIBE_DB_DIR

from lib.data_utils.kp_utils import convert_kps
from lib.data_utils.img_utils import normalize_2d_kp, split_into_chunks, transfrom_keypoints



def keypoint_loss(body_pose, betas, pred_cam, gt_j2d, smpl):
    pred_output = smpl(
            betas=betas,
            body_pose=body_pose[:,1:],
            global_orient=body_pose[:, 0:1],
        )
    pred_joints = pred_output.joints
    pred_keypoints_2d = projection(pred_joints, pred_cam)
    conf = gt_j2d[:,:, -1].unsqueeze(-1).clone()
#     loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_j2d[:, :, :-1])).mean()
    diff = pred_keypoints_2d -  gt_j2d[:, :, :-1]
    loss = (conf * diff.pow(2)).sum()/pred_keypoints_2d.shape[0]
    return loss, pred_keypoints_2ds

if __name__ == "__main__":
    cfg, cfg_file = parse_args()
    eval_name = cfg.TRAIN.get('DATASET_EVAL')
    print('...Evaluating on {} test set...'.format(eval_name))

    model = VIBE(
        n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        seqlen=cfg.DATASET.SEQLEN,
        hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
        pretrained=cfg.TRAIN.PRETRAINED_REGRESSOR,
        add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
        bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
        use_residual=cfg.MODEL.TGRU.RESIDUAL,
    ).to(cfg.DEVICE)

    if cfg.TRAIN.PRETRAINED != '' and os.path.isfile(cfg.TRAIN.PRETRAINED):
        checkpoint = torch.load(cfg.TRAIN.PRETRAINED)
        best_performance = checkpoint['performance']
        model.load_state_dict(checkpoint['gen_state_dict'])

        print(f'==> Loaded pretrained model from {cfg.TRAIN.PRETRAINED}...')
        print(f'Performance on 3DPW test set {best_performance}')
    else:
        print(f'{cfg.TRAIN.PRETRAINED} is not a pretrained model!!!!')
        exit()

    dtype = torch.float
    smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=50,
            create_transl=False,
            dtype = dtype
        ).to(cfg.DEVICE)

    J_regressor = torch.from_numpy(np.load(osp.join(VIBE_DATA_DIR, 'J_regressor_h36m.npy'))).float()
    t_total = 16

    ################## 3dpw ##################
    # dataset_setting = "test"
    # dataset_3dpw = joblib.load("/hdd/zen/data/video_pose/vibe_db/3dpw_{}_db.pt".format(dataset_setting))
    # vid_names = dataset_3dpw['vid_name']
    # thetas = dataset_3dpw['pose']
    # features = dataset_3dpw['features']

    # dataset_3dpw_save = {}

    # with torch.no_grad():
    #     for idx in tqdm(range(len(np.unique(vid_names)))):
    #         curr_name = np.unique(vid_names)[idx]
    #         vid_idxes = np.where(vid_names == curr_name)
    #         curr_thetas = thetas[vid_idxes]
    #         curr_feats = torch.tensor(features[vid_idxes]).to(cfg.DEVICE)


    #         feats_chunks = torch.split(curr_feats, t_total, dim=0)
    #         vibe_acc = []

    #         for feats_chunk in feats_chunks:
    #             preds = model(feats_chunk[None, ], J_regressor=J_regressor)
    #             vibe_acc.append(preds[-1]['theta'][:,:,3:75].cpu().numpy().squeeze())
    #         vibe_acc = np.vstack(vibe_acc)
    #         dataset_3dpw_save[curr_name]  = {
    #             'target_traj' : curr_thetas, 
    #             'feat': curr_feats.cpu().numpy(), 
    #             'traj': vibe_acc
    #         }
    # joblib.dump(dataset_3dpw_save, "/hdd/zen/data/ActBound/AMASS/3dpw_{}_res.pkl".format(dataset_setting))
    ################## 3dpw ##################


    ################## Insta Var ##################
    # h5_file = osp.join(VIBE_DB_DIR, 'insta_train_db.h5')
    # dataset_insta_save = {}
    # with h5py.File(h5_file, 'r') as db:
    #     features = np.array(db['features'])
    #     vid_names = np.array(db['vid_name'])
    #     kp_2ds = np.array(db['joints2D'])
    #     unique_names = np.unique(vid_names)

    #     kp_2ds = convert_kps(kp_2ds, src='insta', dst='spin')
    #     kp_2d_tensor = np.ones((kp_2ds.shape[0], 49, 3), dtype=np.float16)
    #     for idx in range(kp_2ds.shape[0]):
    #         kp_2ds[idx,:,:2] = normalize_2d_kp(kp_2ds[idx,:,:2], 224)
    #         kp_2d_tensor[idx] = kp_2ds[idx]

    #     with torch.no_grad():
    #         for idx in tqdm(range(len(unique_names))):
    #             curr_name = unique_names[idx]
    #             vid_idxes = np.where(vid_names == curr_name)
    #             curr_feats = torch.tensor(features[vid_idxes]).to(cfg.DEVICE)
    #             kp_2d_tensor_curr = kp_2d_tensor[vid_idxes]
                
    #             feats_chunks = torch.split(curr_feats, t_total, dim=0)
    #             vibe_theta_acc = []

    #             for feats_chunk in feats_chunks:
    #                 preds = model(feats_chunk[None, ], J_regressor=J_regressor)
    #                 vibe_theta_acc.append(preds[-1]['theta'].cpu().numpy().squeeze())

    #             vibe_theta_acc = np.vstack(vibe_theta_acc)
    #             dataset_insta_save[curr_name]  = {
    #                 'feat': curr_feats.cpu().numpy(), 
    #                 'vibe_theta': vibe_theta_acc,
    #                 'joints2D': kp_2d_tensor_curr
    #             }

    #     joblib.dump(dataset_insta_save, "/hdd/zen/data/ActBound/AMASS/insta_res.pkl")
    ################## Insta Var ##################

    ################## PennAction ##################
    dataset_name = 'pennaction'
    db = PennAction(seqlen=cfg.DATASET.SEQLEN, debug=False)

    test_loader = DataLoader(
        dataset=db,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    import matplotlib.pyplot as plt
    for i in test_loader:
        kp_2d = i['kp_2d']
        # video = i['video']
        # import pdb
        # pdb.set_trace()
        # print(kp_2d)
        # plt.scatter(kp_2d[0,0,:,0], kp_2d[0, 0, :,1])
        # plt.show()
    exit(0)

    # db_file = osp.join(VIBE_DB_DIR, '{}_train_db.pt').format(dataset_name)
    # dataset_db = joblib.load(db_file)
    # vid_names = dataset_db['vid_name']
    # bboxes = dataset_db['bbox']
    # features = dataset_db['features']
    # kp_2ds = dataset_db['joints2D']
    
    # if dataset_name != 'posetrack':
    #     kp_2ds = convert_kps(kp_2ds, src=dataset_name, dst='spin')

    # kp_2d_tensor = np.ones((kp_2ds.shape[0], 49, 3), dtype=np.float16)

    # for idx in range(kp_2ds.shape[0]):
    #     # crop image and transform 2d keypoints
    #     # print(kp_2ds[idx])
    #     kp_2ds[idx,:,:2], trans = transfrom_keypoints(
    #         kp_2d=kp_2ds[idx,:,:2],
    #         center_x=bboxes[idx,0],
    #         center_y=bboxes[idx,1],
    #         width=bboxes[idx,2],
    #         height=bboxes[idx,3],
    #         patch_width=224,
    #         patch_height=224,
    #         do_augment=False,
    #     )

    #     kp_2ds[idx,:,:2] = normalize_2d_kp(kp_2ds[idx,:,:2], 224)
    #     kp_2d_tensor[idx] = kp_2ds[idx]


    # unique_names = np.unique(vid_names)

    # dataset_save = {}

    
    # with torch.no_grad():
    #     for idx in tqdm(range(len(unique_names))):
    #         curr_name = unique_names[idx]
    #         vid_idxes = np.where(vid_names == curr_name)[0]
    #         curr_feats = torch.tensor(features[vid_idxes]).to(cfg.DEVICE)
    #         kp_2d_tensor_curr = kp_2d_tensor[vid_idxes]
            
    #         feats_chunks = torch.split(curr_feats, t_total, dim=0)
    #         vibe_theta_acc = []

    #         for feats_chunk in feats_chunks:
    #             preds = model(feats_chunk[None, ], J_regressor=J_regressor)
    #             vibe_theta_acc.append(preds[-1]['theta'].cpu().numpy().squeeze())

    #         vibe_theta_acc = np.vstack(vibe_theta_acc)
    #         dataset_save[curr_name]  = {
    #             'feat': curr_feats.cpu().numpy(), 
    #             'vibe_theta': vibe_theta_acc,
    #             'joints2D': kp_2d_tensor_curr
    #         }

    #         if idx > 100:
    #             break
    # joblib.dump(dataset_save, "/hdd/zen/data/ActBound/AMASS/{}_res.pkl".format(dataset_name))