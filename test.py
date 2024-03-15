from PIL import Image
from dataset import SARDataset
from glob import glob
from model import WrapperModel
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset,DataLoader
from tqdm.auto import tqdm
import albumentations as A
import ipdb
import numpy as np
import os
import pathlib
import rasterio
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import ttach as tta

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

def main():
    # normalization
    NORMALIZATION=True
    # TTA
    TTA=True
    transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 90,180,270]),
        #tta.Scale(scales=[1, 2, 4]),
        #tta.Multiply(factors=[0.9, 1, 1.1]),        
    ]
    )

    output_path='./submit/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    #unet_pp=smp.create_model(arch='unetplusplus',classes=1,in_channels=6,encoder_name='timm-resnest269e', encoder_weights="imagenet")
    #model1=WrapperModel.load_from_checkpoint('/home/syo/work/2024_IEEE_GRSS/src/weights/test/epoch=795-step=187060.ckpt',model=unet_pp,mode='test',map_location=torch.device("cuda"))
    #model1.eval()

    unet_pp=smp.create_model(arch='unetplusplus',classes=1,in_channels=6,encoder_name='tu-maxvit_large_tf_512', encoder_weights=None)
    model1=WrapperModel.load_from_checkpoint('/home/syo/work/2024_IEEE_GRSS/src/weights/maxvit-800e/epoch=791-step=72864.ckpt',model=unet_pp,mode='test',map_location=torch.device("cuda"))
    model1.eval()
    if TTA:
        model1 = tta.SegmentationTTAWrapper(model1, transforms)


    image_root=pathlib.Path('/home/syo/work/2024_IEEE_GRSS/dataset/Track1/val/images/')
    #image_root=pathlib.Path('/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/images/')
    images_list = sorted(list(image_root.glob('*')))
    thred=0.1

    for i in tqdm(images_list):
        sar_data = rasterio.open(i).read()
        if NORMALIZATION:
            sar_data=_sar_normalization(sar_data=sar_data)
        with torch.no_grad():
                sar_data=torch.tensor(sar_data,dtype=torch.float32).unsqueeze(0).cuda()
                outputs1=model1.forward(sar_data)
                outputs1 = F.sigmoid(outputs1)
                outputs=outputs1
                #outputs2=model2.forward(sar_data)
                #outputs2 = F.sigmoid(outputs2)

                #outputs3=model3.forward(sar_data)
                #outputs3 = F.sigmoid(outputs3)

                #outputs=0.34*outputs1+0.34*outputs2+0.32*outputs3
                preds=outputs.detach().cpu().numpy().squeeze(0)

        binary_map = (preds > thred).astype(np.uint8)[0]
        nid=i.stem
        bmap_img=Image.fromarray(binary_map,'L')
        bmap_img.save(output_path+nid+'.png')

if __name__ == "__main__":
    main()
