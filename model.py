import torch
from torch import nn
import lightning as L
import torchvision
import timm
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from transformers import get_cosine_schedule_with_warmup
from transformers import get_polynomial_decay_schedule_with_warmup
from torch import optim
import ipdb
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


def calculate_f1_score(predicted_map, ground_truth_map):
    # Flatten the maps to 1D arrays
    predicted_flat = predicted_map.flatten()
    ground_truth_flat = ground_truth_map.flatten()

    # Calculate F1 score
    f1 = f1_score(ground_truth_flat, predicted_flat)

    return f1

class WrapperModel(L.LightningModule):
    def __init__(self, model, mode='train',learning_rate=1e-5,train_dataloader=None, val_dataloader=None) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn=nn.BCEWithLogitsLoss()
        if mode=='test':
            return
        self.train_dl = train_dataloader
        self.train_ds = train_dataloader.dataset
        self.valid_ds = val_dataloader.dataset

    def forward(self, input):
        output = self.model(input)
        return output

    def training_step(self, i):
        input = i["data"]
        label = i["label"]
        output = self.forward(input)
        loss = self.loss_fn(input=output, target=label)
        self.log("train/loss", loss)
        # self.log("lr",self.lr_schedulers().get_lr()[0])
        return loss

    def configure_optimizers(self):
        steps_per_ep = len(self.train_dl)
        train_steps = len(self.train_dl) * self.trainer.max_epochs  # max epouch 100
        # optimizer = optim.AdamW(self.parameters(), lr=1e-5)
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate,betas=(0.9, 0.999), weight_decay=0.05)
        #lr_scheduler = get_cosine_schedule_with_warmup(
        #    optimizer,
        #    num_warmup_steps=int(steps_per_ep * self.trainer.max_epochs * 0.03),
        #    num_training_steps=train_steps,
        #)
        lr_scheduler= get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,num_warmup_steps=int(steps_per_ep*self.trainer.max_epochs * 0.03),
            num_training_steps=train_steps,
            power=3)
        return [optimizer], [
            {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        ]

    def validation_step(self, i):
        input = i["data"]
        label = i["label"]
        output = self.forward(input)
        loss = self.loss_fn(input=output, target=label)
        self.log("train/valid_loss", loss)

    def get_score(self, thred=0.5,type="valid"):
        preds = []
        labels = []
        if type == "valid":
            ds = self.valid_ds
        elif type == "train":
            ds = self.train_ds

        with torch.no_grad():
            for batch in ds:
                data = batch["data"].unsqueeze(0).cuda()
                outputs = self.forward(data)
                outputs = F.sigmoid(outputs)
                preds.append(outputs.detach().cpu().numpy())
                labels.append(batch['label'].unsqueeze(0).detach().cpu().numpy())
        preds = np.vstack(preds)
        labels = np.vstack(labels)
        binary_map = (preds > thred).astype(int)
        precision, recall, f1_score, _ = precision_recall_fscore_support(labels.flatten(), binary_map.flatten(), average='binary')
        return precision, recall, f1_score

    def on_validation_epoch_end(self):
        step=19
        if self.current_epoch % step == step-1:
        #if self.current_epoch >= self.trainer.max_epochs -1:
        #if True:
            precision, recall, f1_score = self.get_score(type="valid")
            self.log("train/valid_pr", precision)
            self.log("train/valid_re", recall)
            self.log("train/valid_f1", f1_score)
            precision, recall, f1_score = self.get_score(type="train")
            self.log("train/train_pr", precision)
            self.log("train/train_re", recall)
            self.log("train/train_f1", f1_score)
        else:
            pass
