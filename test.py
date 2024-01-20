import torch
from torch.utils.data import Dataset,DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
import ipdb
import rasterio
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from model import WrapperModel
from dataset import SARDataset
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import pathlib
import os
from torch import nn

def main1():
    output_path='./submit/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    #unet_pp=smp.create_model(arch='unetplusplus',classes=2,in_channels=6)
    #unet_pp=smp.create_model(arch='unetplusplus',classes=2,in_channels=6,encoder_name='resnet101', encoder_weights="imagenet") 
    #model=WrapperModel.load_from_checkpoint('/home/syo/epoch=105-step=8692.ckpt',model=unet_pp,mode='test')
    unet_pp=smp.create_model(arch='unetplusplus',classes=2,in_channels=6,encoder_name='timm-resnest101e', encoder_weights="imagenet")
    model=WrapperModel.load_from_checkpoint('/home/syo/epoch=78-step=8611.ckpt',model=unet_pp,mode='test')
    #model=WrapperModel.load_from_checkpoint('/home/syo/work/2024_IEEE_GRSS/src/waterflow/unet_pp_1/version_0/checkpoints/epoch=29-step=2460.ckpt',model=unet_pp,mode='test')
    model.eval()


    image_root=pathlib.Path('/home/syo/work/2024_IEEE_GRSS/dataset/Track1/val/images/')
    #image_root=pathlib.Path('/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/images/')
    images_list = sorted(list(image_root.glob('*')))
    thred=0.5

    for i in tqdm(images_list):
        sar_data = rasterio.open(i).read()
        sar_data=torch.tensor(sar_data,dtype=torch.float32).unsqueeze(0).cuda()
        with torch.no_grad():
            outputs=model.forward(sar_data)
            outputs = F.softmax(outputs)
            preds=outputs.detach().cpu().numpy()
        binary_map = (preds[:,1,:,:] > thred).astype(np.uint8)[0]
        nid=i.stem
        bmap_img=Image.fromarray(binary_map,'L')
        bmap_img.save(output_path+nid+'.png')

def main():
    output_path='./submit/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    unet_pp=smp.create_model(arch='unetplusplus',classes=2,in_channels=6,encoder_name='resnet101', encoder_weights="imagenet") 
    model=WrapperModel.load_from_checkpoint('/home/syo/epoch=105-step=8692.ckpt',model=unet_pp,mode='test')
    model.eval()

    train_dataset=SARDataset(
        image_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/images/",
        label_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/labels/",
        mode='train'
    )

    for batch in tqdm(train_dataset):
        data = batch["data"].unsqueeze(0).cuda()
        label=batch['label'].unsqueeze(0).cuda()
        with torch.no_grad():
            outputs = model.forward(data)
            loss = nn.CrossEntropyLoss(reduction='sum')(input=outputs, target=label)
            ipdb.set_trace()
            print(label)
            print(loss)
        # binary_map = (preds[:,1,:,:] > thred).astype(int)[0]

if __name__ == "__main__":
    main1()