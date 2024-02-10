import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import ipdb
import os
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn
import torch

import lightning as L
import timm
from timm.models.registry import model_entrypoint
import pickle
import torch.nn.functional as F

from dataset import build_dataloader,build_dataloader_w_pseudo
from model import WrapperModel
from lightning.pytorch.loggers import WandbLogger

from lightning.pytorch.callbacks import LearningRateMonitor,StochasticWeightAveraging
import torchvision
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
import segmentation_models_pytorch as smp
from lightning.pytorch.loggers import TensorBoardLogger


def main():
    debug=False
    train_data_aug=True
    val_size=0.2
    if train_data_aug:
        #data_transforms = {
        #    "train": A.Compose(
        #        [
        #            A.HorizontalFlip(p=0.5),
        #            A.VerticalFlip(p=0.5),
        #            A.GridDistortion(p=0.2),
        #            A.RandomResizedCrop(p=0.2,width=512,height=512),
        #            A.OneOf(
        #                [
        #                    A.GaussianBlur(),
        #                    A.GaussNoise(var_limit=(0,2e-8),mean=0,per_channel=True),
        #                ],
        #                p=0.2,
        #            ),
        #            #ToTensorV2(),
        #        ],
        #        p=1.0,
        #    ),
        #}
        data_transforms = {
        "train": A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.OneOf(
                    [
                        A.Rotate(limit=(90,90)),
                        A.Rotate(limit=(-90,-90)),
                        A.Rotate(limit=(180,180)),
                        A.Rotate(limit=(-180,-180)),
                    ],
                    p=0.5,
                ),
                A.OneOf([A.RandomResizedCrop(width=512,height=512),A.RandomGridShuffle()],p=0.3),
                A.OneOf([A.GridDistortion(), A.OpticalDistortion()], p=0.3),
                A.OneOf(
                    [
                        A.GaussianBlur(),
                        A.GaussNoise(var_limit=(0, 2e-8), mean=0, per_channel=True),
                    ],
                    p=0.2,
                ),
                # ToTensorV2(),
            ],
            p=1.0,
        ),
    }

        train_dataloader, val_dataloader = build_dataloader_w_pseudo(batch_size=16,num_workers=4,val_size=0.05,seed=42,data_transforms=data_transforms['train'])
    else:
        train_dataloader, val_dataloader = build_dataloader_w_pseudo(batch_size=16,num_workers=4,val_size=0.05,seed=42)

    #unet_pp=smp.create_model(arch='unetplusplus',classes=2,in_channels=6, encoder_weights="imagenet")
    #unet_pp=smp.create_model(arch='unetplusplus',classes=2,in_channels=6,encoder_name='resnet101', encoder_weights="imagenet")
    #unet_pp=smp.create_model(arch='unetplusplus',classes=2,in_channels=6,encoder_name='timm-resnest101e', encoder_weights="imagenet")
    unet_pp=smp.create_model(arch='unetplusplus',classes=2,in_channels=6,encoder_name='timm-efficientnet-b4', encoder_weights="imagenet")
    #unet_pp=smp.create_model(arch='unetplusplus',classes=2,in_channels=6,encoder_name='timm-resnest269e', encoder_weights="imagenet")
    #unet_pp=smp.create_model(arch='unetplusplus',classes=2,in_channels=6,encoder_name='timm-efficientnet-b3', encoder_weights="imagenet")
    #unet_pp=smp.create_model(arch='pspnet',classes=2,in_channels=6,encoder_name='timm-resnest101e', encoder_weights="imagenet")
    wrapper_model = WrapperModel(model=unet_pp,train_dataloader=train_dataloader,val_dataloader=val_dataloader,learning_rate=3e-5)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    #checkpoint_callback = ModelCheckpoint(monitor='train/valid_loss',save_top_k=3)
    checkpoint_callback = ModelCheckpoint(save_top_k=-1,every_n_epochs=19)

    if not debug:
        logger = WandbLogger(project="waterflow", name="unet_pp_1")
    else:
        logger = TensorBoardLogger("waterflow", name="unet_pp_1")
        
    trainer = L.Trainer(max_epochs=100, precision="bf16", logger=logger, callbacks=[lr_monitor,checkpoint_callback],log_every_n_steps=1)
    
    trainer.fit( model=wrapper_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
