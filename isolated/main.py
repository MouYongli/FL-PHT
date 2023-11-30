import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import NibabelDataset
from lr_scheduler import PolyLRScheduler
from losses import DC_and_CE_loss, MemoryEfficientSoftDiceLoss
from losses import get_tp_fp_fn_tn

from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor

def numpy_collate(batch):
    batch = torch.utils.data._utils.collate.default_collate(batch)
    for k in batch.keys():
        batch[k] = batch[k].numpy()
    return batch

def collate_outputs(outputs: List[dict]):
    """
    used to collate default train_step and validation_step outputs. If you want something different then you gotta
    extend this
    we expect outputs to be a list of dictionaries where each of the dict has the same set of keys
    """
    collated = {}
    for k in outputs[0].keys():
        if np.isscalar(outputs[0][k]):
            collated[k] = [o[k] for o in outputs]
        elif isinstance(outputs[0][k], np.ndarray):
            collated[k] = np.vstack([o[k][None] for o in outputs])
        elif isinstance(outputs[0][k], list):
            collated[k] = [item for o in outputs for item in o[k]]
        else:
            raise ValueError(f'Cannot collate input of type {type(outputs[0][k])}. '
                             f'Modify collate_outputs to add this functionality')
    return collated

if __name__ == '__main__':
    # Dataset and Dataloader
    BATCH_SIZE = 2
    NUM_EPOCHS = 1
    LR = 1e-2
    WEIGHT_DECAY = 3e-5
    PATCH_SIZE = [128, 128, 128]
    DEVICE = torch.device(type='cuda', index=0) if torch.cuda.is_available() else torch.device(type='cpu')

    dset_train = NibabelDataset()
    dset_val = NibabelDataset()
    dloader_train = DataLoader(dset_train, BATCH_SIZE, collate_fn=numpy_collate)
    dloader_val = DataLoader(dset_val, BATCH_SIZE, collate_fn=numpy_collate)

    # Data Transforms
    transforms_train = []
    transforms_train.append(SpatialTransform(
            PATCH_SIZE, patch_center_dist_from_border=None,
            do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
            do_rotation=False, # do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
            p_rot_per_axis=1,  # todo experiment with this
            do_scale=True, scale=(0.7, 1.4),
            border_mode_data="constant", border_cval_data=0, order_data=3,
            border_mode_seg="constant", border_cval_seg=-1, order_seg=1,
            random_crop=False,  # random cropping is part of our dataloaders
            p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False  # todo experiment with this
    ))
    transforms_train.append(GaussianNoiseTransform(p_per_sample=0.1))
    transforms_train.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5))
    transforms_train.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    transforms_train.append(ContrastAugmentationTransform(p_per_sample=0.15))
    transforms_train.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5, order_downsample=0, order_upsample=3, p_per_sample=0.25, ignore_axes=None))
    transforms_train.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    transforms_train.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))
    transforms_train.append(RemoveLabelTransform(-1, 0))
    transforms_train.append(RenameTransform('seg', 'target', True))
    transforms_train.append(NumpyToTensor(['data', 'target'], 'float'))
    transforms_train = Compose(transforms_train)

    transforms_val = []
    transforms_val.append(GaussianNoiseTransform(p_per_sample=0.1))
    transforms_val.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5))
    transforms_val.append(RenameTransform('seg', 'target', True))
    transforms_val.append(NumpyToTensor(['data', 'target'], 'float'))
    transforms_val = Compose(transforms_val)

    # Model
    model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4, (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=False).to(DEVICE)

    # Loss function
    soft_dice_kwargs = {
        'batch_dice': True,
        'smooth': 1e-5,
        'do_bg': False,
        'ddp': False
    }
    ce_kwargs = {}
    loss_func = DC_and_CE_loss(soft_dice_kwargs, ce_kwargs, dice_class=MemoryEfficientSoftDiceLoss)

    # Optimizer and LR_Scheduler
    optimizer = torch.optim.SGD(model.parameters(), LR, weight_decay= WEIGHT_DECAY, momentum=0.99, nesterov=True)
    lr_scheduler = PolyLRScheduler(optimizer, LR, NUM_EPOCHS)

    # Log files

    # Training start
    print(f"Training starts...")
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch} starts...")
        # Training Epoch
        model.train()
        lr_scheduler.step(epoch)
        train_outputs = []
        for i, batch in enumerate(dloader_train):
            batch = transforms_train(**batch)
            data, target = batch["data"], batch["target"]
            data = data.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            optimizer.step()
            train_outputs.append({'loss': loss.detach().cpu().numpy()})
            print()
        train_collate_outputs = collate_outputs(train_outputs)
        train_loss = np.mean(train_collate_outputs['loss'])
        print('     train_losses', train_loss, epoch)
        # Val Epoch
        model.eval()
        val_outputs = []
        for i, batch in enumerate(dloader_val):
            batch = transforms_train(**batch)
            data, target = batch["data"], batch["target"]
            data = data.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            output = model(data)
            loss = loss_func(output, target)
            output_seg = output.argmax(1)[:, None]
            preds_onehot_test = torch.zeros(output.shape, device=DEVICE, dtype=torch.float32)
            preds_onehot_test.scatter_(1, output_seg, 1)
            tp, fp, fn, tn = get_tp_fp_fn_tn(preds_onehot_test, target)
            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()
            val_outputs.append({'loss': loss.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard})
        val_collate_outputs = collate_outputs(val_outputs)
        tp_sum = np.sum(val_collate_outputs['tp_hard'], 0)
        fp_sum = np.sum(val_collate_outputs['fp_hard'], 0)
        fn_sum = np.sum(val_collate_outputs['fn_hard'], 0)
        val_loss = np.mean(val_collate_outputs['loss'])
        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp_sum, fp_sum, fn_sum)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        print('     val_losses', val_loss, epoch)
        print('     val mean_fg_dice', mean_fg_dice, epoch)
        print('     val dice_per_class_or_region', global_dc_per_class, epoch)