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
import random
from lightning.pytorch.callbacks import RichModelSummary
from lightning.pytorch.callbacks import RichProgressBar

import hydra
from omegaconf import DictConfig, OmegaConf

from glob import glob

from sklearn.model_selection import KFold 
from dataset import SARDataset

import wandb
##============
##mmseg
#import mmseg
#from mmseg.registry import MODELS
#import mmengine
#from mmengine import Config
#from mmseg.utils import register_all_modules
#register_all_modules()


#==========
def main():
    debug=False
    val_size=0.1
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
                #A.OneOf(
                #    [
                #        A.GaussianBlur(),
                #        A.GaussNoise(var_limit=(0, 2e-8), mean=0, per_channel=True),
                #    ],
                #    p=0.2,
                #),
                # ToTensorV2(),
            ],
            p=1.0,
        ),
    }

    train_dataloader, val_dataloader = build_dataloader(batch_size=8,num_workers=8,val_size=0.1,seed=42,data_transforms=data_transforms['train'])

    unet_pp=smp.create_model(arch='unetplusplus',classes=1,in_channels=6,encoder_name='timm-resnest269e', encoder_weights="imagenet")
    wrapper_model = WrapperModel(model=unet_pp,train_dataloader=train_dataloader,val_dataloader=val_dataloader,learning_rate=1e-4)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress_bar = RichProgressBar()
    model_summary = RichModelSummary(max_depth=3)
    loss_checkpoint_callback = ModelCheckpoint(
        verbose=True,
        filename=f"val_loss-" + "epoch_{epoch}-val_loss_{val/loss:.4f}-score_{score/valid_f1:.4f}",
        monitor="valid/loss",
        mode="min",
        save_top_k=5,
        save_last=True,
        save_weights_only=True,
        auto_insert_metric_name=False,
    )

    score_checkpoint_callback = ModelCheckpoint(
        verbose=True,
        filename=f"val_score-" + "epoch_{epoch}-val_loss_{val/loss:.4f}-socre_{score/valid_f1:.4f}",
        monitor="score/train_f1",
        save_top_k=5,
        save_weights_only=True,
        mode="max",
        auto_insert_metric_name=False,
    )

    if not debug:
        logger = WandbLogger(project="waterflow", name="unet_pp_1")
    else:
        logger = TensorBoardLogger("waterflow", name="unet_pp_1")
        
    trainer = L.Trainer(max_epochs=400, precision="bf16-mixed", logger=logger, callbacks=[lr_monitor,loss_checkpoint_callback,score_checkpoint_callback],log_every_n_steps=10,accumulate_grad_batches=1,gradient_clip_val=1)
    
    trainer.fit(model=wrapper_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

def kflod():
    debug=False
    val_size=0.1
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
            ],
            p=1.0,
        ),
    }

    ############ MODEL #####################################
    #unet_pp=smp.create_model(arch='unetplusplus',classes=1,in_channels=6,encoder_name='timm-resnest269e', encoder_weights="imagenet")
    unet_pp=smp.create_model(arch='unetplusplus',classes=1,in_channels=6,encoder_name='tu-maxvit_base_tf_512', encoder_weights="imagenet")
    wrapper_model = WrapperModel(model=unet_pp,learning_rate=1e-4)

    ############ HOOKS ###########################
    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress_bar = RichProgressBar()
    model_summary = RichModelSummary(max_depth=3)
    loss_checkpoint_callback = ModelCheckpoint(
        verbose=True,
        filename=f"val_loss-" + "epoch_{epoch}-val_loss_{valid/loss:.4f}-score_{score/valid_f1:.4f}",
        monitor="valid/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        save_weights_only=True,
        auto_insert_metric_name=False,
    )
    score_checkpoint_callback = ModelCheckpoint(
        verbose=True,
        filename=f"val_score-" + "epoch_{epoch}-val_loss_{valid/loss:.4f}-socre_{score/valid_f1:.4f}",
        monitor="score/train_f1",
        save_top_k=3,
        save_weights_only=True,
        mode="max",
        auto_insert_metric_name=False,
    )

    ##### datalodar, kflod, train #################

    DATASET_ROOT='/home/syo/work/2024_IEEE_GRSS/dataset/'
    TRACK1_ROOT='/home/syo/work/2024_IEEE_GRSS/dataset/Track1/'
    TRACK2_ROOT='/home/syo/work/2024_IEEE_GRSS/dataset/Track2/'    
    images_list = sorted(list(glob(TRACK1_ROOT+'train/images/' + "*")))
    label_list = sorted(list(glob(TRACK1_ROOT+'train/labels/' + "*")))

    kf_func = KFold(
        n_splits=int(1 / 0.1), random_state=42, shuffle=True
    )
    kf_l = kf_func.split(images_list)

    for i, (train_index, val_index) in enumerate(kf_l):
        if not debug:
            #logger = WandbLogger(project="waterflow", name="unet_pp_1")
            logger = WandbLogger(
                project="ema-KFP",
                # log_model="all",
                name=f"KF{i}",
            )
        else:
            logger = TensorBoardLogger("waterflow", name="unet_pp_1")

        train_dataset=SARDataset(
            data=[images_list[i] for i in train_index],
            targets=[label_list[i] for i in train_index],
            data_transforms=None,
        )
        val_dataset=SARDataset(
            data=[images_list[i] for i in val_index],
            targets=[label_list[i] for i in val_index],
            data_transforms=None,
        )

        train_dataloader = DataLoader(train_dataset, batch_size=8, num_workers=16, shuffle=True, pin_memory=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=16, shuffle=False, pin_memory=True)

        trainer = L.Trainer(max_epochs=400, precision="bf16-mixed", logger=logger, callbacks=[lr_monitor,loss_checkpoint_callback,score_checkpoint_callback],log_every_n_steps=10,accumulate_grad_batches=1,gradient_clip_val=1)
        trainer.fit(model=wrapper_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        wandb.finish()

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    seed_everything(42)
    kflod()
