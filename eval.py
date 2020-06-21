import os
import torch

from lib.dataset import *
from lib.models import VIBE
from lib.core.evaluate import Evaluator
from lib.core.config import parse_args
from torch.utils.data import DataLoader

# from kinematic_synthesis.lib.model import get_vae_model, get_smpl_refiner
from kinematic_synthesis.utils.config import Config

def main(cfg):
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

    test_db = eval(eval_name)(set='test', seqlen=cfg.DATASET.SEQLEN, debug=cfg.DEBUG)
    
    test_loader = DataLoader(
        dataset=test_db,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    # reinder_cfg = Config("zen_rf_9",local=True)
    # refiner = get_smpl_refiner(reinder_cfg, cfg.DEVICE, num_vae_epoch= 20)

    Evaluator(
        model=model,
        device=cfg.DEVICE,
        test_loader=test_loader,
        refiner = None
    ).run()


if __name__ == '__main__':
    cfg, cfg_file = parse_args()

    main(cfg)
