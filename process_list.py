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

import os
import os.path as osp
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from lib.models.vibe import VIBE_Demo
from lib.utils.renderer import Renderer
from lib.dataset.inference import Inference
from lib.data_utils.kp_utils import convert_kps
from lib.utils.pose_tracker import run_posetracker

from lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)

MIN_NUM_FRAMES = 25

def main(args):
    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f'Loading video list {args.video_list}')
    video_list = [l.strip() for l in open(args.vid_list, 'r').readlines()]
    if len(video_list) < 1:
        print('No files were found in video list')
        return

    print('Loading VIBE model')
    # ========= Define VIBE model ========= #
    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    # ========= Load VIBE pretrained weights ========= #
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')

    num_videos = len(video_list)
    print(f'Processing {num_videos} videos.')
    for video_idx, video_file in enumerate(video_list, start=1):
        if not osp.isfile(video_file):
            print(f'Input video \"{video_file}\" does not exist! Moving on to next file.')
            continue

        filename = osp.splitext(osp.basename(video_file))[0]
        output_path = osp.join(args.output_folder, filename)
        os.makedirs(output_path, exist_ok=True)

        image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)

        print(f'[{video_idx}/{num_videos}] Processing {num_frames} frames')
        orig_height, orig_width = img_shape[:2]

        # ========= Run tracking ========= #
        bbox_scale = 1.1
        if args.tracking_method == 'pose':
            if not osp.isabs(video_file):
                video_file = osp.join(os.getcwd(), video_file)
            tracking_results = run_posetracker(video_file, staf_folder=args.staf_dir, display=args.display)
        else:
            # run multi object tracker
            mot = MPT(
                device=device,
                batch_size=args.tracker_batch_size,
                display=args.display,
                detector_type=args.detector,
                output_format='dict',
                yolo_img_size=args.yolo_img_size,
            )
            tracking_results = mot(image_folder)

        # remove tracklets if num_frames is less than MIN_NUM_FRAMES
        for person_id in list(tracking_results.keys()):
            if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
                del tracking_results[person_id]

        # ========= Run VIBE on each person ========= #
        print(f'Running VIBE on each tracklet...')
        vibe_results = {}
        for person_id in tqdm(list(tracking_results.keys())):
            bboxes = joints2d = None

            if args.tracking_method == 'bbox':
                bboxes = tracking_results[person_id]['bbox']
            elif args.tracking_method == 'pose':
                joints2d = tracking_results[person_id]['joints2d']

            frames = tracking_results[person_id]['frames']

            dataset = Inference(
                image_folder=image_folder,
                frames=frames,
                bboxes=bboxes,
                joints2d=joints2d,
                scale=bbox_scale,
            )

            bboxes = dataset.bboxes
            frames = dataset.frames
            has_keypoints = True if joints2d is not None else False

            dataloader = DataLoader(dataset, batch_size=args.vibe_batch_size, num_workers=16)

            with torch.no_grad():

                pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

                for batch in dataloader:
                    if has_keypoints:
                        batch, nj2d = batch
                        norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                    batch = batch.unsqueeze(0)
                    batch = batch.to(device)

                    batch_size, seqlen = batch.shape[:2]
                    output = model(batch)[-1]

                    pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                    pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                    pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
                    pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))
                    pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))


                pred_cam = torch.cat(pred_cam, dim=0)
                pred_verts = torch.cat(pred_verts, dim=0)
                pred_pose = torch.cat(pred_pose, dim=0)
                pred_betas = torch.cat(pred_betas, dim=0)
                pred_joints3d = torch.cat(pred_joints3d, dim=0)

                del batch

            # ========= [Optional] run Temporal SMPLify to refine the results ========= #
            if args.run_smplify and args.tracking_method == 'pose':
                norm_joints2d = np.concatenate(norm_joints2d, axis=0)
                norm_joints2d = convert_kps(norm_joints2d, src='staf', dst='spin')
                norm_joints2d = torch.from_numpy(norm_joints2d).float().to(device)

                # Run Temporal SMPLify
                update, new_opt_vertices, new_opt_cam, new_opt_pose, new_opt_betas, \
                new_opt_joints3d, new_opt_joint_loss, opt_joint_loss = smplify_runner(
                    pred_rotmat=pred_pose,
                    pred_betas=pred_betas,
                    pred_cam=pred_cam,
                    j2d=norm_joints2d,
                    device=device,
                    batch_size=norm_joints2d.shape[0],
                    pose2aa=False,
                )

                # update the parameters after refinement
                print(f'Update ratio after Temporal SMPLify: {update.sum()} / {norm_joints2d.shape[0]}')
                pred_verts = pred_verts.cpu()
                pred_cam = pred_cam.cpu()
                pred_pose = pred_pose.cpu()
                pred_betas = pred_betas.cpu()
                pred_joints3d = pred_joints3d.cpu()
                pred_verts[update] = new_opt_vertices[update]
                pred_cam[update] = new_opt_cam[update]
                pred_pose[update] = new_opt_pose[update]
                pred_betas[update] = new_opt_betas[update]
                pred_joints3d[update] = new_opt_joints3d[update]

            elif args.run_smplify and args.tracking_method == 'bbox':
                print('[WARNING] You need to enable pose tracking to run Temporal SMPLify algorithm!')
                print('[WARNING] Continuing without running Temporal SMPLify!..')

            # ========= Save results to a pickle file ========= #
            pred_cam = pred_cam.cpu().numpy()
            pred_verts = pred_verts.cpu().numpy()
            pred_pose = pred_pose.cpu().numpy()
            pred_betas = pred_betas.cpu().numpy()
            pred_joints3d = pred_joints3d.cpu().numpy()

            orig_cam = convert_crop_cam_to_orig_img(
                cam=pred_cam,
                bbox=bboxes,
                img_width=orig_width,
                img_height=orig_height
            )

            output_dict = {
                'pred_cam': pred_cam,
                'orig_cam': orig_cam,
                'verts': pred_verts,
                'pose': pred_pose,
                'betas': pred_betas,
                'joints3d': pred_joints3d,
                'joints2d': joints2d,
                'bboxes': bboxes,
                'frame_ids': frames,
            }

            vibe_results[person_id] = output_dict

        # Clean-up the temporal folder
        shutil.rmtree(image_folder)

        # Save the outputs to joblib pkl file. File is loaded through joblib.load(pkl_path)
        output_pkl_path = osp.join(args.output_folder, f'{filename}.pkl')
        print(f'Saving output results to \"{output_pkl_path}\".')
        joblib.dump(vibe_results, output_pkl_path)

    # Clean-up after processing
    del model

    print('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_list', type=str,
                        help='input list with paths of videos')

    parser.add_argument('--output_folder', type=str,
                        help='output folder to write results')

    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking')

    parser.add_argument('--staf_dir', type=str, default='/home/mkocabas/developments/openposetrack',
                        help='path to directory STAF pose tracking method installed.')

    parser.add_argument('--vibe_batch_size', type=int, default=450,
                        help='batch size of VIBE')

    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')

    parser.add_argument('--run_smplify', action='store_true',
                        help='run smplify for refining the results, you need pose tracking to enable it')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')

    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='process the video list with the specified GPU id')

    args = parser.parse_args()

    main(args)
