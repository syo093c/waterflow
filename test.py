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
import albumentations as A

def tta(model,input_data):
    """
    input_data: batch_size, channel, w, h
    """
    data_transforms = {
        "horizion": A.HorizontalFlip(p=1),
        "vertical": A.VerticalFlip(p=1),
    }
    A.HorizontalFlip()


def _sar_normalization(sar_data):
    """
        normalize the data to (0, 1).

        Band 1: SAR image, VV
        Band 2: SAR image, VH
        Band 3: Merit DEM
        Band 4: Copernicus DEM
        Band 5: ESA World Cover Map
        Band 6: Water occurrence probability

        # 1: 0-32765        x/32765
        # 2:                x/32765
        # 3: -9999:9999     (x+9999)/(9999*2)
        # 4: 0-100          x/100
        # 5: 0-255          x/255
    """
    sar_data=sar_data.astype(np.float32)
    sar_data[0]=sar_data[0]/32765
    sar_data[1]=sar_data[1]/32765
    sar_data[2]=(sar_data[2]+9999)/(9999*2)
    sar_data[3]=(sar_data[3]+9999)/(9999*2)
    sar_data[4]=sar_data[4]/100
    sar_data[5]=sar_data[5]/255

    return sar_data
def main1():
    # normalization
    NORMALIZATION=True

    output_path='./submit/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    #unet_pp=smp.create_model(arch='unetplusplus',classes=2,in_channels=6)
    #unet_pp=smp.create_model(arch='unetplusplus',classes=2,in_channels=6,encoder_name='resnet101', encoder_weights="imagenet") 
    #model=WrapperModel.load_from_checkpoint('/home/syo/epoch=105-step=8692.ckpt',model=unet_pp,mode='test')
    #unet_pp=smp.create_model(arch='unetplusplus',classes=2,in_channels=6,encoder_name='timm-resnest101e', encoder_weights="imagenet")
    #model=WrapperModel.load_from_checkpoint('/home/syo/epoch=78-step=8611.ckpt',model=unet_pp,mode='test')

    unet_pp=smp.create_model(arch='unetplusplus',classes=2,in_channels=6,encoder_name='timm-resnest269e', encoder_weights="imagenet")
    model1=WrapperModel.load_from_checkpoint('/home/syo/work/2024_IEEE_GRSS/src/waterflow/h8z2cs4g/checkpoints/epoch=299-step=48900.ckpt',model=unet_pp,mode='test',map_location=torch.device("cuda"))
    model1.eval()

    #unet_pp=smp.create_model(arch='unetplusplus',classes=2,in_channels=6,encoder_name='timm-efficientnet-b8', encoder_weights="imagenet")
    #model2=WrapperModel.load_from_checkpoint('/home/syo/work/2024_IEEE_GRSS/src/weights/efnet-epoch=299-step=48900.ckpt',model=unet_pp,mode='test')
    ##model=WrapperModel.load_from_checkpoint('/home/syo/work/2024_IEEE_GRSS/src/waterflow/unet_pp_1/version_0/checkpoints/epoch=29-step=2460.ckpt',model=unet_pp,mode='test')
    #model2.eval()

    #unet_pp=smp.create_model(arch='unetplusplus',classes=2,in_channels=6,encoder_name='timm-resnest269e', encoder_weights="imagenet")
    #model3=WrapperModel.load_from_checkpoint('/home/syo/work/2024_IEEE_GRSS/src/weights/aug/resnest-aug-epoch=299-step=48900.ckpt',model=unet_pp,mode='test',map_location=torch.device("cuda"))
    #model3.eval()

    image_root=pathlib.Path('/home/syo/work/2024_IEEE_GRSS/dataset/Track1/val/images/')
    #image_root=pathlib.Path('/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/images/')
    images_list = sorted(list(image_root.glob('*')))
    thred=0.5

    for i in tqdm(images_list):
        sar_data = rasterio.open(i).read()
        if NORMALIZATION:
            sar_data=_sar_normalization(sar_data=sar_data)
        sar_data=torch.tensor(sar_data,dtype=torch.float32).unsqueeze(0).cuda()
        with torch.no_grad():
            ipdb.set_trace()
            outputs1=model1.forward(sar_data)
            outputs1 = F.softmax(outputs1)
            outputs=outputs1

            #outputs2=model2.forward(sar_data)
            #outputs2 = F.softmax(outputs2)

            #outputs3=model3.forward(sar_data)
            #outputs3 = F.softmax(outputs3)

            #outputs=0.34*outputs1+0.34*outputs2+0.32*outputs3
            preds=outputs.detach().cpu().numpy()
        binary_map = (preds[:,1,:,:] > thred).astype(np.uint8)[0]
        nid=i.stem
        bmap_img=Image.fromarray(binary_map,'L')
        bmap_img.save(output_path+nid+'.png')

#def main():
#    output_path='./submit/'
#    if not os.path.exists(output_path):
#        os.mkdir(output_path)
#
#    unet_pp=smp.create_model(arch='unetplusplus',classes=2,in_channels=6,encoder_name='resnet101', encoder_weights="imagenet") 
#    model=WrapperModel.load_from_checkpoint('/home/syo/work/2024_IEEE_GRSS/src/waterflow/y0c9tylo/checkpoints/epoch=299-step=55200.ckpt',model=unet_pp,mode='test')
#    model.eval()
#
#    train_dataset=SARDataset(
#        image_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/images/",
#        label_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/labels/",
#        mode='train'
#    )
#
#    for batch in tqdm(train_dataset):
#        data = batch["data"].unsqueeze(0).cuda()
#        label=batch['label'].unsqueeze(0).cuda()
#        with torch.no_grad():
#            outputs = model.forward(data)
#            loss = nn.CrossEntropyLoss(reduction='sum')(input=outputs, target=label)
#            ipdb.set_trace()
#            print(label)
#            print(loss)
#        # binary_map = (preds[:,1,:,:] > thred).astype(int)[0]

if __name__ == "__main__":
    main1()
