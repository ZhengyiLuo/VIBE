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

import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp
from progress.bar import Bar

from lib.core.config import VIBE_DATA_DIR
from lib.utils.utils import move_dict_to_device, AverageMeter
import pickle as pk

from lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
)

# import zen_renderer.renderer.smpl_renderer as smpl_renderer
logger = logging.getLogger(__name__)

class DataSaver():
    def __init__(
            self,
            test_loader,
            model,
            device=None,
            refiner = None,
    ):
        self.test_loader = test_loader
        self.model = model
        self.device = device

        self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts'])
        self.refiner = refiner
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def validate(self):
        self.model.eval()

        start = time.time()

        summary_string = ''

        bar = Bar('Validation', fill='#', max=len(self.test_loader))

        if self.evaluation_accumulators is not None:
            for k,v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []

        J_regressor = torch.from_numpy(np.load(osp.join(VIBE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

        preds_acc = []
        targets_acc = []
        for i, target in enumerate(self.test_loader):

            # video = video.to(self.device)
            move_dict_to_device(target, self.device)

            # <=============
            with torch.no_grad():
                inp = target['features']
                
                # preds = self.model(inp, J_regressor=J_regressor, refiner = self.refiner)
                preds = self.model(inp, J_regressor=J_regressor, refiner = None)

                # convert to 14 keypoint format for evaluation
                # if self.use_spin:
                n_kp = preds[-1]['kp_3d'].shape[-2]
                pred_j3d = preds[-1]['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
                target_j3d = target['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
                pred_verts = preds[-1]['verts'].view(-1, 6890, 3).cpu().numpy()
                target_theta = target['theta'].view(-1, 85).cpu().numpy()

                self.evaluation_accumulators['pred_verts'].append(pred_verts)
                self.evaluation_accumulators['target_theta'].append(target_theta)

                self.evaluation_accumulators['pred_j3d'].append(pred_j3d)
                self.evaluation_accumulators['target_j3d'].append(target_j3d)

                ######################## Data saver #####################
                preds_acc.append({k:v.cpu().numpy() if torch.is_tensor(v) else v for k, v in preds[-1].items() \
                    if k in ['features', 'theta', 'instance_id']})
                targets_acc.append({k:v.cpu().numpy() if torch.is_tensor(v) else v for k, v in target.items() \
                     if k in ['features', 'theta', 'instance_id']})
                
                # renderer = smpl_renderer.SMPL_Renderer(image_size = 400, camera_mode="look_at")
                # target_pose = target_theta[:,3:75]
                # pred_pose = preds[-1]['theta'][:,:,3:75].squeeze()
                # renderer.render_pose_vid(torch.tensor(target_pose), out_file_name = "output/gt{:02d}.mp4".format(i), random_camera = 2, random_shape=False)
                # renderer.render_pose_vid(torch.tensor(pred_pose), out_file_name = "output/ref{:02d}.mp4".format(i), random_camera = 2, random_shape=False)

                ######################## vis #####################
                del target, preds
                torch.cuda.empty_cache()
            # =============>

            batch_time = time.time() - start

            summary_string = f'({i + 1}/{len(self.test_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                             f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'

            bar.suffix = summary_string
            bar.next()

        bar.finish()

        logger.info(summary_string)

        vibe_3dpw_res = {
            "preds": preds_acc,
            "targets": targets_acc,
        }
        # pk.dump(vibe_3dpw_res, open("3dpw_vibe_res.pkl", "wb"))
        # pk.dump(vibe_3dpw_res, open("/hdd/zen/data/ActBound/AMASS/amass_vibe_train_res.pkl", "wb"))
        # pk.dump(vibe_3dpw_res, open("/hdd/zen/data/ActBound/AMASS/3dpw_train_res.pkl", "wb"))
        # pk.dump(vibe_3dpw_res, open("/hdd/zen/data/ActBound/AMASS/3dpw_test_res.pkl", "wb"))
    

    def evaluate(self):

        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k] = np.vstack(v)

        pred_j3ds = self.evaluation_accumulators['pred_j3d']
        target_j3ds = self.evaluation_accumulators['target_j3d']

        pred_j3ds = torch.from_numpy(pred_j3ds).float()
        target_j3ds = torch.from_numpy(target_j3ds).float()

        print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
        pred_pelvis = (pred_j3ds[:,[2],:] + pred_j3ds[:,[3],:]) / 2.0
        target_pelvis = (target_j3ds[:,[2],:] + target_j3ds[:,[3],:]) / 2.0


        pred_j3ds -= pred_pelvis
        target_j3ds -= target_pelvis

        # Absolute error (MPJPE)
        errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
        errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        pred_verts = self.evaluation_accumulators['pred_verts']
        target_theta = self.evaluation_accumulators['target_theta']

        # pk.dump(target_theta, open("3dpw_thetas.pkl", "wb"))
        # np.savez("amass_eval_test.npz", pred_j3ds = pred_j3ds.numpy(), target_j3ds = target_j3ds.numpy())
        
        m2mm = 1000

        pve = np.mean(compute_error_verts(target_theta=target_theta, pred_verts=pred_verts)) * m2mm
        accel = np.mean(compute_accel(pred_j3ds)) * m2mm
        accel_err = np.mean(compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)) * m2mm
        mpjpe = np.mean(errors) * m2mm
        pa_mpjpe = np.mean(errors_pa) * m2mm

        eval_dict = {
            'mpjpe': mpjpe,
            'pa-mpjpe': pa_mpjpe,
            'pve': pve,
            'accel': accel,
            'accel_err': accel_err
        }

        log_str = ' '.join([f'{k.upper()}: {v:.4f},'for k,v in eval_dict.items()])
        print(log_str)

    def run(self):
        self.validate()
        self.evaluate()