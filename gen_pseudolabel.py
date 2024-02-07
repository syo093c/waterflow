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

import ipdb

def _trans_vec(vec):
    vec=np.transpose(vec, (2, 0, 1))
    return vec

def tta(model,input_data):
    """
    input_data: batch_size, channel, w, h
    """

    tta_out=[]
    input_data=np.transpose(input_data, (1, 2, 0))

    data_transforms = {
        "horizon": A.Compose([A.HorizontalFlip(p=1,),]),
        "vertical": A.Compose([A.VerticalFlip(p=1)]),
        "r90": A.Compose([A.Rotate(limit=(90,90),p=1)]),
        "rn90": A.Compose([A.Rotate(limit=(-90,-90),p=1)]),
        "rn180": A.Compose([A.Rotate(limit=(-180,-180),p=1)]),
    }

    horizon_fliped =    _trans_vec(data_transforms["horizon"](image=input_data)["image"])
    vertical_fliped =   _trans_vec(data_transforms["vertical"](image=input_data)["image"])
    r90 =               _trans_vec(data_transforms["r90"](image=input_data)["image"])
    rn90 =              _trans_vec(data_transforms["rn90"](image=input_data)["image"])
    rn180 =             _trans_vec(data_transforms["rn180"](image=input_data)["image"])
    raw_input =        _trans_vec(input_data)

    # raw
    input=torch.tensor(raw_input,dtype=torch.float32).unsqueeze(0).cuda()
    output=model.forward(input)
    output = F.softmax(output).squeeze(0)
    preds=output.detach().cpu().numpy()
    tta_out.append(preds)

    # horizon
    input=torch.tensor(horizon_fliped,dtype=torch.float32).unsqueeze(0).cuda()
    output=model.forward(input)
    output = F.softmax(output).squeeze(0)
    preds=output.detach().cpu().numpy()
    preds=np.transpose(preds, (1, 2, 0))    
    preds=_trans_vec(data_transforms['horizon'](image=preds)['image'])
    tta_out.append(preds)

    # vertical
    input=torch.tensor(vertical_fliped,dtype=torch.float32).unsqueeze(0).cuda()
    output=model.forward(input)
    output = F.softmax(output).squeeze(0)
    preds=output.detach().cpu().numpy()
    preds=np.transpose(preds, (1, 2, 0))    
    preds=_trans_vec(data_transforms['vertical'](image=preds)['image'])
    tta_out.append(preds)

    # 90
    input=torch.tensor(r90,dtype=torch.float32).unsqueeze(0).cuda()
    output=model.forward(input)
    output = F.softmax(output).squeeze(0)
    preds=output.detach().cpu().numpy()
    preds=np.transpose(preds, (1, 2, 0))    
    preds=_trans_vec(data_transforms['rn90'](image=preds)['image'])
    tta_out.append(preds)

    # -90
    input=torch.tensor(rn90,dtype=torch.float32).unsqueeze(0).cuda()
    output=model.forward(input)
    output = F.softmax(output).squeeze(0)
    preds=output.detach().cpu().numpy()
    preds=np.transpose(preds, (1, 2, 0))    
    preds=_trans_vec(data_transforms['r90'](image=preds)['image'])
    tta_out.append(preds)

    # -180
    input=torch.tensor(rn180,dtype=torch.float32).unsqueeze(0).cuda()
    output=model.forward(input)
    output = F.softmax(output).squeeze(0)
    preds=output.detach().cpu().numpy()
    preds=np.transpose(preds, (1, 2, 0))    
    preds=_trans_vec(data_transforms['rn180'](image=preds)['image'])
    tta_out.append(preds)

    # mean preds
    mean_pred = np.mean(tta_out, axis=0)
    return torch.tensor(mean_pred).unsqueeze(0).numpy()


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

    output_path='./val_pseudo_label/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    unet_pp=smp.create_model(arch='unetplusplus',classes=2,in_channels=6,encoder_name='timm-resnest269e', encoder_weights="imagenet")
    model1=WrapperModel.load_from_checkpoint('/home/syo/work/2024_IEEE_GRSS/src/weights/nor-aug3-unetpp-800e-timm-resnest269e-server2/epoch=795-step=146464.ckpt',model=unet_pp,mode='test',map_location=torch.device("cuda"))
    model1.eval()

    image_root=pathlib.Path('/home/syo/work/2024_IEEE_GRSS/dataset/Track1/val/images/')
    #image_root=pathlib.Path('/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/images/')
    images_list = sorted(list(image_root.glob('*')))
    thred=0.47

    for i in tqdm(images_list):
        sar_data = rasterio.open(i).read()
        if NORMALIZATION:
            sar_data=_sar_normalization(sar_data=sar_data)
        with torch.no_grad():
            if not TTA:
                sar_data=torch.tensor(sar_data,dtype=torch.float32).unsqueeze(0).cuda()
                outputs1=model1.forward(sar_data)
                outputs1 = F.softmax(outputs1)
                outputs=outputs1
                #outputs2=model2.forward(sar_data)
                #outputs2 = F.softmax(outputs2)

                #outputs3=model3.forward(sar_data)
                #outputs3 = F.softmax(outputs3)

                #outputs=0.34*outputs1+0.34*outputs2+0.32*outputs3
                preds=outputs.detach().cpu().numpy()
            else:
                preds1=tta(model=model1,input_data=sar_data)
                preds=preds1
                #preds2=tta(model=model2,input_data=sar_data)
                #preds3=tta(model=model3,input_data=sar_data)
                #preds=0.50*preds1+0.05*preds2+0.45*preds3

        binary_map = (preds[:,1,:,:] > thred).astype(np.uint8)[0]
        nid=i.stem
        bmap_img=Image.fromarray(binary_map,'L')
        bmap_img.save(output_path+nid+'.png')


if __name__ == "__main__":
    main()
