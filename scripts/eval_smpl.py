import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch
import pickle as pk
import numpy as np

from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS
from lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
from lib.core.config import VIBE_DATA_DIR
from lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
)
from zen_renderer.utils.transform_utils import (
    convert_aa_to_orth6d, convert_orth_6d_to_aa, vertizalize_smpl_root, convert_orth_6d_to_mat
)


def smpl_to_joints(input_pose, smpl, J_regressor = None):
    # import pdb       
    # pdb.set_trace()
    curr_batch_size = input_pose.shape[0]
    dtype = input_pose.dtype
    device = input_pose.device

    input_pose_aa = convert_orth_6d_to_aa(input_pose)
    input_pose_vertical = vertizalize_smpl_root(input_pose_aa)
    input_pose_6d = convert_aa_to_orth6d(torch.tensor(input_pose_vertical)).reshape(curr_batch_size, -1)
    input_pose = torch.tensor(input_pose_6d).to(input_pose.device).float()
    
#     pred_rotmat = rot6d_to_rotmat(input_pose).view(curr_batch_size, 24, 3, 3).to(device)
    pred_rotmat = convert_orth_6d_to_mat(input_pose).to(device)
    
    betas = torch.zeros(curr_batch_size, 10, dtype = dtype).to(device)
    pred_output = smpl(betas=betas,body_pose=pred_rotmat[:, 1:], global_orient= pred_rotmat[:, 0].unsqueeze(1),pose2rot=False)

    pred_vertices = pred_output.vertices
    pred_joints = pred_output.joints
    if J_regressor is not None:
        J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
        pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
        pred_joints = pred_joints[:, H36M_TO_J14, :]

    # pred_keypoints_2d = projection(pred_joints, pred_cam)
    
    pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

    output = {
        # 'theta'  : torch.cat([pred_cam, pose, pred_shape], dim=1),
        'verts'  : pred_vertices,
        # 'kp_2d'  : pred_keypoints_2d,
        'kp_3d'  : pred_joints,
        'rotmat' : pred_rotmat
    }
    return output

def compute_metric_on_seqs(seq_pred, seq_gt, smpl, J_regressor):
    output_pred = smpl_to_joints(seq_pred, smpl, J_regressor)
    output_gt = smpl_to_joints(seq_gt, smpl, J_regressor)
    metrics = compute_metric_on_outputs(output_gt=output_gt, output_pred=output_pred)
    return metrics

def compute_metric_on_outputs(output_gt, output_pred):
    m2mm = 1000
    verts_gt = output_gt['verts']
    pt3d_gt = output_gt['kp_3d']

    verts_pred = output_pred['verts']
    pt3d_pred = output_pred['kp_3d']
    
    pred_pelvis = (pt3d_pred[:,[2],:] + pt3d_pred[:,[3],:]) / 2.0
    target_pelvis = (pt3d_gt[:,[2],:] + pt3d_gt[:,[3],:]) / 2.0
    pt3d_pred -= pred_pelvis
    pt3d_gt -= target_pelvis

    errors = torch.sqrt(((pt3d_gt - pt3d_pred) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    # S1_hat = batch_compute_similarity_transform_torch(pt3d_pred, pt3d_gt)
    # errors_pa = torch.sqrt(((S1_hat - pt3d_gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

    pve = np.mean(compute_error_verts(verts_pred.cpu().numpy(), target_verts=verts_gt.cpu().numpy())) * m2mm
    accel = np.mean(compute_accel(pt3d_pred.cpu().numpy())) * m2mm
    accel_err = np.mean(compute_error_accel(joints_pred=pt3d_pred.cpu().numpy(), joints_gt=pt3d_gt.cpu().numpy())) * m2mm
    mpjpe = np.mean(errors) * m2mm
    
    # pa_mpjpe = np.mean(errors_pa) * m2mm
    return [mpjpe, pve, accel_err]


if __name__ == "__main__":
    dtype = torch.float32

    vae_res_path = "/hdd/zen/data/ActBound/AMASS/real_fake_train_take2.pkl"
    vae_ress = pk.load(open(vae_res_path, "rb"))
    
    

    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    device = torch.device("cpu")
    smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=150,
            create_transl=False,
            dtype = dtype
        ).to(device)
    J_regressor = torch.from_numpy(np.load(osp.join(VIBE_DATA_DIR, 'J_regressor_h36m.npy'))).float()
    m2mm = 1000
    print(VIBE_DATA_DIR)
    ################## Rendering #################
    results = []
    with torch.no_grad():
        for idx, (k, v) in enumerate(vae_ress.items()):
            gt_pose, flip_pose = v['pose'], v['flip_pose']
            gt_pose, flip_pose = torch.tensor(gt_pose, dtype = dtype).to(device), torch.tensor(flip_pose, dtype = dtype).to(device)

            if flip_pose.shape[0] > 0:
                eval_res = compute_metric_on_seqs(gt_pose, flip_pose[0], smpl, J_regressor)
                print(eval_res)
                results.append(eval_res)
            # break
    import pdb       
    pdb.set_trace()