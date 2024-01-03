import json
import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
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

import mlflow

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

def nearest_multiple_of_32(x):
    remainder = x % 32
    if remainder < 16:
        return x - remainder
    else:
        return x + (32 - remainder)
    
def nearest_multiple_of_64(x):
    remainder = x % 64
    if remainder < 32:
        return x - remainder
    else:
        return x + (64 - remainder)
    
class Trainer():
    def __init__(
            self,
            training_batch_size: int = 1,
            validation_batch_size: int = 1,
            epochs: int = 5,
            learning_rate: float = 0.001,
            weight_decay: float = 1e-2,
            patch_size = [128, 128, 128],
            device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            client_id: int = 0,
            training_set_path: str = '/home/data/MICCAI_FeTS2022_TrainingData',
            validation_set_path: str = '/home/data/MICCAI_FeTS2022_ValidationData',
            training_set_partition_file: str = 'partitioning_1.csv',
            validation_set_partition_file: str = 'partitioning_1.csv'
    ):
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patch_size = patch_size
        self.device = device
        self.client_id = client_id

        # Data Loaders
        dset_train = NibabelDataset(base_path=training_set_path, partition_id=client_id, partitioning=training_set_partition_file)
        dset_val = NibabelDataset(base_path=validation_set_path, partition_id=client_id, partitioning=validation_set_partition_file)
        self.dloader_train = DataLoader(dset_train, self.training_batch_size, collate_fn=numpy_collate)
        self.dloader_val = DataLoader(dset_val, self.validation_batch_size, collate_fn=numpy_collate)

        # Data Transforms
        transforms_train = []
        transforms_train.append(SpatialTransform(
                self.patch_size, patch_center_dist_from_border=None,
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
        self.transforms_train = transforms_train

        transforms_val = []
        transforms_val.append(RemoveLabelTransform(-1, 0))
        transforms_val.append(RenameTransform('seg', 'target', True))
        transforms_val.append(NumpyToTensor(['data', 'target'], 'float'))
        transforms_val = Compose(transforms_val)
        self.transforms_val = transforms_val

        self.model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4, (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=False).to(device)

        # Loss function
        soft_dice_kwargs = {
            'batch_dice': True,
            'smooth': 1e-5,
            'do_bg': False,
            'ddp': False
        }
        ce_kwargs = {}
        self.loss_func = DC_and_CE_loss(soft_dice_kwargs, ce_kwargs, dice_class=MemoryEfficientSoftDiceLoss)

        # Optimizer and LR_Scheduler
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.learning_rate, weight_decay= self.weight_decay, momentum=0.99, nesterov=True)
        self.lr_scheduler = PolyLRScheduler(self.optimizer, self.learning_rate, self.epochs)

    def run_training(self):
        print(f"Training starts for client {self.client_id} on device {self.device}...")
        path = str(os.environ.get('FEDERATED_MODEL_PATH'))
        modelName = 'model.pth'
        fedmodelpath = path + '/' + modelName

        # Load model if exists
        if os.path.exists(fedmodelpath):
            print('Model exists, loading model')
            self.model.load_state_dict(torch.load(fedmodelpath))

        train_losses = []
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}/{self.epochs} starts...")
            # Training Epoch
            self.model.train()
            self.lr_scheduler.step(epoch)
            train_outputs = []
            for i, batch in enumerate(self.dloader_train):
                batch = self.transforms_train(**batch)
                data, target = batch["data"], batch["target"]
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                output = self.model(data)
                loss = self.loss_func(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.optimizer.step()
                train_outputs.append({'loss': loss.detach().cpu().numpy()})
                print(f'{i}/{len(self.dloader_train)} loss:', loss.detach().cpu().numpy())
            train_collate_outputs = collate_outputs(train_outputs)
            train_loss = np.mean(train_collate_outputs['loss'])
            print('     train_losses', train_loss, epoch)
            train_losses.append(train_loss)
            
        # Do validation at the end of training round
        self.model.eval()
        val_outputs = []
        for i, batch in enumerate(self.dloader_val):
            batch = self.transforms_val(**batch)
            data, target = batch["data"], batch["target"]
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            n, m, d, h, w = data.shape
            # d_new, h_new, w_new = nearest_multiple_of_32(d), nearest_multiple_of_32(h), nearest_multiple_of_32(w)
            d_new, h_new, w_new = nearest_multiple_of_64(d), nearest_multiple_of_64(h), nearest_multiple_of_64(w)
            data_interpolated = F.interpolate(data, size=(d_new//2, h_new//2, w_new//2), mode='trilinear', align_corners=False)
            # print(data.shape, data_interpolated.shape)
            output = self.model(data_interpolated)            
            output_interpolated = F.interpolate(output, size=(d, h, w), mode='trilinear', align_corners=False)
            loss = self.loss_func(output_interpolated, target)
            output_seg = output_interpolated.argmax(1)[:, None]
            preds_onehot_test = torch.zeros(output_interpolated.shape, device=self.device, dtype=torch.float32)
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
        
        # Flatten the list and convert numpy.float32 to float
        val_dice_per_class_or_region = [float(x) for sublist in global_dc_per_class for x in sublist]
        train_losses = [float(x) for x in train_losses]

        loss_file_contents = {
            'train_losses': train_losses,
            'client_id': self.client_id,
            'val_loss': float(val_loss),
            'val_mean_fg_dice': float(mean_fg_dice),
            'val_dice_per_class_or_region': val_dice_per_class_or_region
        }   
        # Save losses
        loss_file = 'losses.json'
        loss_path = path + '/' + loss_file
        # Save losses to file
        with open(loss_path, 'w') as f:
            json.dump(loss_file_contents, f)
        # Save model
        torch.save(self.model.state_dict(), fedmodelpath)