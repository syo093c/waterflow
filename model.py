import lightning as L
import torchvision
import timm
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from transformers import get_cosine_schedule_with_warmup
from transformers import get_polynomial_decay_schedule_with_warmup
from torch import optim
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
import torch
from segmentation_models_pytorch.losses import DiceLoss
from torchmetrics import F1Score
import gc
from ema import EMAOptimizer

def calculate_f1_score(predicted_map, ground_truth_map):
    # Flatten the maps to 1D arrays
    predicted_flat = predicted_map.flatten()
    ground_truth_flat = ground_truth_map.flatten()

    # Calculate F1 score
    f1 = f1_score(ground_truth_flat, predicted_flat)

    return f1

class WrapperModel(L.LightningModule):
    def __init__(self, model, mode='train',learning_rate=3e-5) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.bce_logit_loss=nn.BCEWithLogitsLoss()
        self.dice_loss=DiceLoss(mode='binary',from_logits=True)
        if mode=='test':
            return

        self.train_pred=[]
        self.val_pred=[]
        self.train_label=[]
        self.val_label=[]

        self.f1=F1Score(task="binary")

    def forward(self, input):
        output = self.model(input)
        return output

    def training_step(self, i):
        input = i["data"]
        label = i["label"]
        output = self.forward(input)
        bce_loss = self.bce_logit_loss(input=output, target=label)
        dice_loss = self.dice_loss(y_pred=output, y_true=label)
        loss=0.25*bce_loss+0.75*dice_loss
        self.log("train/loss", loss)
        self.log("train/bce_loss", bce_loss)
        self.log("train/dice_loss", dice_loss)
        # self.log("lr",self.lr_schedulers().get_lr()[0])

        self.train_pred.append(output.detach().cpu())
        self.train_label.append(label.detach().cpu())

        return loss
        #if self.trainer.current_epoch < self.trainer.max_epochs * 0.4:
        #    return bce_loss
        #else:
        #    return loss

    def configure_optimizers(self):
        steps_per_ep = len(self.train_dl)
        train_steps = len(self.train_dl) * self.trainer.max_epochs  # max epouch 100
        # optimizer = optim.AdamW(self.parameters(), lr=1e-5)
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate,betas=(0.9, 0.999), weight_decay=0.05)
        optimizer= EMAOptimizer(optimizer=optimizer,device=torch.device('cuda'))
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(steps_per_ep * self.trainer.max_epochs * 0.03/self.trainer.accumulate_grad_batches),
            num_training_steps=int(train_steps/self.trainer.accumulate_grad_batches),
        )
        #lr_scheduler= get_polynomial_decay_schedule_with_warmup(
        #    optimizer=optimizer,num_warmup_steps=int(steps_per_ep*self.trainer.max_epochs * 0.03),
        #    num_training_steps=train_steps,
        #    power=3)
        return [optimizer], [
            {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        ]

    def validation_step(self, i):
        input = i["data"]
        label = i["label"]
        output = self.forward(input)

        bce_loss = self.bce_logit_loss(input=output, target=label)
        dice_loss = self.dice_loss(y_pred=output, y_true=label)
        loss=0.25*bce_loss+0.75*dice_loss
        self.log("valid/loss", loss)
        self.log("valid/bce_loss", bce_loss)
        self.log("valid/dice_loss", dice_loss)

        self.val_pred.append(output.detach().cpu())
        self.val_label.append(label.detach().cpu())
    
    def on_train_epoch_end(self):
        train_pred = (torch.cat(self.train_pred).sigmoid()>0.5).float()
        train_label = torch.cat(self.train_label)
        self.train_pred.clear()
        self.train_label.clear()

        f1_score = self.f1(train_pred,train_label)
        gc.collect()
        self.log("score/train_f1", f1_score)

    def on_validation_epoch_end(self):
        val_pred = (torch.cat(self.val_pred).sigmoid()>0.5).float()
        val_label = torch.cat(self.val_label)
        self.val_pred.clear()
        self.val_label.clear()

        f1_score = self.f1(val_pred,val_label)
        gc.collect()
        self.log("score/valid_f1", f1_score)
