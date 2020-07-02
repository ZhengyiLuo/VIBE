import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F

from lib.core.config import VIBE_DATA_DIR
from lib.models.spin import Regressor, hmr
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS
from zen_renderer.utils.transform_utils import convert_orth_6d_to_mat, convert_aa_to_orth6d
from kinematic_synthesis.utils.config import Config
from kinematic_synthesis.lib.model import *
from lib.utils.geometry import rotation_matrix_to_angle_axis



def projection(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
                                               translation=pred_cam_t,
                                               focal_length=5000.,
                                               camera_center=camera_center)
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
    return pred_keypoints_2d


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

class REFINERV2(nn.Module):
    def __init__(
            self,
            vibe,
            cfg = "zen_rec_23"
    ):
        super(REFINERV2, self).__init__()

        vae_cfg = Config(cfg)
        self.vae_model, _, _ = get_models(vae_cfg, iter = -1)
        self.vibe = vibe
        self.vibe.eval()

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False
        )

    # def parameters(self):
    #     return self.refiner_model.parameters()

    def state_dict(self):   
        return {
            "refiner": self.vae_model.state_dict(), 
            "vibe": self.vibe.state_dict()
        }

    def load_state_dict(self, input):   
        self.vae_model.load_state_dict(input['refiner'])
        self.vibe.load_state_dict(input['vibe'])

    def train(self):
        self.vae_model.train()
        self.vibe.train()

    def eval(self):
        self.vae_model.eval()
        self.vibe.eval()

    def forward(self, input, J_regressor=None):
        # input size NTF
        batch_size, seqlen = input.shape[:2]

        vibe_output = self.vibe(input)[-1]

        vibe_theta = vibe_output['theta']
        vibe_pose = vibe_theta[:,:,3:75]
        pred_cam = vibe_theta[:,:,:3].reshape(batch_size * seqlen, 3)
        pred_shape = vibe_theta[:,:,75:].reshape(batch_size * seqlen, 10)
        vibe_pose_6d = convert_aa_to_orth6d(vibe_pose).reshape(vibe_pose.shape[0], vibe_pose.shape[1], -1)

        X_r= self.vae_model(vibe_pose_6d.permute(1, 0, 2), input.permute(1, 0, 2))
        # X_r= self.refiner_model(input.permute(1, 0, 2))
        X_r = X_r.permute(1, 0, 2).contiguous()
        
        pred_rotmat = convert_orth_6d_to_mat(X_r).reshape(batch_size * seqlen, 24, 3, 3)
        # pred_rotmat = convert_orth_6d_to_mat(vibe_pose_6d).reshape(batch_size * seqlen, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0:1],
            pose2rot=False
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        if J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            pred_joints = pred_joints[:, H36M_TO_J14, :]

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        smpl_output = [{
            'theta'  : torch.cat([pred_cam, pose, pred_shape], dim=1),
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,
            'rotmat' : pred_rotmat
        }]

        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

        return smpl_output

class REFINERV2_demo(REFINERV2):
    def __init__(
            self,
            vibe,
            cfg = "zen_vis_5", 
            pretrained=osp.join(VIBE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):
        super().__init__(vibe, cfg)
        self.hmr = hmr()
        checkpoint = torch.load(pretrained)
        self.hmr.load_state_dict(checkpoint['model'], strict=False)

    def forward(self, input, J_regressor=None):
        batch_size, seqlen, nc, h, w = input.shape

        feature = self.hmr.feature_extractor(input.reshape(-1, nc, h, w))

        input = feature.reshape(batch_size, seqlen, -1)

        # input size NTF
        batch_size, seqlen = input.shape[:2]

        vibe_output = self.vibe(input)[-1]

        vibe_theta = vibe_output['theta']
        vibe_pose = vibe_theta[:,:,3:75]
        pred_cam = vibe_theta[:,:,:3].reshape(batch_size * seqlen, 3)
        pred_shape = vibe_theta[:,:,75:].reshape(batch_size * seqlen, 10)
        vibe_pose_6d = convert_aa_to_orth6d(vibe_pose).reshape(vibe_pose.shape[0], vibe_pose.shape[1], -1)

        X_r= self.refiner_model(vibe_pose_6d.permute(1, 0, 2), input.permute(1, 0, 2))
        X_r = X_r.permute(1, 0, 2).contiguous()[:,:seqlen, :]
        
        pred_rotmat = convert_orth_6d_to_mat(X_r).reshape(batch_size * seqlen, 24, 3, 3)
        # pred_rotmat = convert_orth_6d_to_mat(vibe_pose_6d).reshape(batch_size * seqlen, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0:1],
            pose2rot=False
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        if J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            pred_joints = pred_joints[:, H36M_TO_J14, :]

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        smpl_output = [{
            'theta'  : torch.cat([pred_cam, pose, pred_shape], dim=1),
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,
            'rotmat' : pred_rotmat
        }]

        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

        return smpl_output


if __name__ == "__main__":
    from kinematic_synthesis.utils.config import Config
    from kinematic_synthesis.lib.model import *
    from lib.dataset import *
    from torch.utils.data import DataLoader
    
    meva_model = MEVA(90)
    db = PennAction(seqlen=90, debug=False)

    test_loader = DataLoader(
        dataset=db,
        batch_size=32,
        shuffle=False,
        num_workers=1,
    )

    for i in test_loader:
        kp_2d = i['kp_2d']
        input = i['features']
        meva_model(input)
