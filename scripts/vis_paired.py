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
from lib.models import VIBE, MEVAV2
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14
from lib.core.config import parse_args
from torch.utils.data import DataLoader
from lib.core.config import VIBE_DATA_DIR, VIBE_DB_DIR

from lib.data_utils.kp_utils import convert_kps
from lib.data_utils.img_utils import normalize_2d_kp, split_into_chunks, transfrom_keypoints
from copycat.smpllib.smpl_mujoco import SMPL_M_Renderer
from zen_renderer.utils.transform_utils import vertizalize_smpl_root
from zen_renderer.utils.image_utils import assemble_videos

if __name__ == "__main__":
    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    vibe_model = VIBE(
        n_layers=2,
        batch_size=32,
        seqlen=16,
        hidden_size=1024,
        add_linear=True,
        bidirectional=False,
        use_residual=True,
    ).to(device)
    vibe_dir = 'data/vibe_data/vibe_model_wo_3dpw.pth.tar'
    checkpoint = torch.load(vibe_dir)
    best_performance = checkpoint['performance']
    vibe_model.load_state_dict(checkpoint['gen_state_dict'])
    vibe_model.eval()

    print(f'==> Loaded pretrained model from {vibe_dir}...')
    print(f'Performance on 3DPW test set {best_performance}')

    meva_model = MEVAV2(
        90, hidden_size=1024, add_linear=True, use_residual=True, bidirectional=True, n_layers=2, batch_size=32
    ).to(device)
    meva_dir = 'results/meva/30-06-2020_13-35-09_meva/model_best.pth.tar'
    checkpoint = torch.load(meva_dir)
    best_performance = checkpoint['performance']
    meva_model.load_state_dict(checkpoint['gen_state_dict'])
    meva_model.eval()
    print(f'==> Loaded pretrained model from {meva_dir}...')
    print(f'Performance on 3DPW test set {best_performance}')

    dtype = torch.float
    image_size = 400
    renderer = SMPL_M_Renderer(render_size = (image_size, image_size))
    output_base = "/hdd/zen/data/ActmixGenenerator/output/3dpw"
    output_path = osp.join(output_base, "mevav2")
    if not osp.isdir(output_path): os.makedirs(output_path)
    J_regressor = torch.from_numpy(np.load(osp.join(VIBE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

    ################## 3dpw ##################
    t_total = 90
    dataset_setting = "test"
    dataset_3dpw = joblib.load("/hdd/zen/data/video_pose/vibe_db/3dpw_{}_db.pt".format(dataset_setting))
    vid_names = dataset_3dpw['vid_name']
    thetas = dataset_3dpw['pose']
    features = dataset_3dpw['features']
    dataset_3dpw_save = {}
    with torch.no_grad():
        for idx in tqdm(range(len(np.unique(vid_names)))):
            curr_name = np.unique(vid_names)[idx]
            vid_idxes = np.where(vid_names == curr_name)
            mocap_acc = thetas[vid_idxes]
            curr_feats = torch.tensor(features[vid_idxes]).to(device)

            feats_chunks = torch.split(curr_feats, t_total, dim=0)
            vibe_acc = []
            meva_acc= []

            for feats_chunk in feats_chunks:
                vibe_preds = vibe_model(feats_chunk[None, ])
                vibe_acc.append(vibe_preds[-1]['theta'][0,:,3:75].cpu().numpy())
                meva_preds = meva_model(feats_chunk[None, ])
                meva_acc.append(meva_preds[-1]['theta'][0,:,3:75].cpu().numpy())
                
                
            vibe_acc = np.vstack(vibe_acc)
            meva_acc = np.vstack(meva_acc)

            mocap_pose = vertizalize_smpl_root(torch.tensor(mocap_acc)).numpy()
            vibe_pose = vertizalize_smpl_root(torch.tensor(vibe_acc)).cpu().numpy()
            meva_pose = vertizalize_smpl_root(torch.tensor(meva_acc)).numpy()
            
            mocap_images = renderer.render_smpl(mocap_pose)
            vibe_images = renderer.render_smpl(vibe_pose)
            meva_images = renderer.render_smpl(meva_pose)

            videos = [mocap_images, vibe_images, meva_images]
            grid_size = [1,len(videos)]
            descriptions = ["Mocap", "VIBE", "MEVA"]
            output_name = "{}/output_meva{:02d}.mp4".format(output_path, idx)
            assemble_videos(videos, grid_size, descriptions, output_name)
            print(output_name)


    ################## 3dpw ##################
    