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

    unet_pp=smp.create_model(arch='unetplusplus',classes=1,in_channels=6,encoder_name='timm-resnest269e', encoder_weights="imagenet")
    m1=smp.create_model(arch='unetplusplus',classes=1,in_channels=6,encoder_name='tu-maxvit_base_tf_512', encoder_weights=None)
    m2=smp.create_model(arch='unetplusplus',classes=1,in_channels=6,encoder_name='tu-maxvit_base_tf_512', encoder_weights=None)
    m3=smp.create_model(arch='unetplusplus',classes=1,in_channels=6,encoder_name='tu-maxvit_base_tf_512', encoder_weights=None)
    m4=smp.create_model(arch='unetplusplus',classes=1,in_channels=6,encoder_name='tu-maxvit_base_tf_512', encoder_weights=None)

    PATH1='/home/syo/work/2024_IEEE_GRSS/src/weights/kfold/nybho6h3/val_loss-epoch_370-val_loss_0.1409-score_0.9035.ckpt'
    PATH2='/home/syo/work/2024_IEEE_GRSS/src/weights/kfold/pouoecch/val_loss-epoch_341-val_loss_0.1199-score_0.9385.ckpt'
    PATH3='/home/syo/work/2024_IEEE_GRSS/src/weights/kfold/qaedzm0a/val_loss-epoch_344-val_loss_0.1344-score_0.9170.ckpt'
    PATH4='/home/syo/work/2024_IEEE_GRSS/src/weights/kfold/vw5mt1dv/val_score-epoch_391-val_loss_0.1095-socre_0.9429.ckpt'

    model1=WrapperModel.load_from_checkpoint(PATH1,model=m1,mode='test',map_location=torch.device("cuda"))
    model2=WrapperModel.load_from_checkpoint(PATH2,model=m2,mode='test',map_location=torch.device("cuda"))
    model3=WrapperModel.load_from_checkpoint(PATH3,model=m3,mode='test',map_location=torch.device("cuda"))
    model4=WrapperModel.load_from_checkpoint(PATH4,model=m4,mode='test',map_location=torch.device("cuda"))
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()

    if TTA:
        model1 = tta.SegmentationTTAWrapper(model1, transforms)
        model2 = tta.SegmentationTTAWrapper(model2, transforms)
        model3 = tta.SegmentationTTAWrapper(model3, transforms)
        model4 = tta.SegmentationTTAWrapper(model4, transforms)


    image_root=pathlib.Path('/home/syo/work/2024_IEEE_GRSS/dataset/Track1/val/images/')
    images_list = sorted(list(image_root.glob('*')))
    thred=0.47

    for i in tqdm(images_list):
        sar_data = rasterio.open(i).read()
        if NORMALIZATION:
            sar_data=_sar_normalization(sar_data=sar_data)
        with torch.no_grad():
                sar_data=torch.tensor(sar_data,dtype=torch.float32).unsqueeze(0).cuda()
                outputs1=model1.forward(sar_data)
                outputs1 = F.sigmoid(outputs1)
                outputs2=model2.forward(sar_data)
                outputs2 = F.sigmoid(outputs2)
                outputs3=model3.forward(sar_data)
                outputs3 = F.sigmoid(outputs3)
                outputs4=model4.forward(sar_data)
                outputs4 = F.sigmoid(outputs4)

                outputs=0.25*outputs1+0.25*outputs2+0.25*outputs3+0.25*outputs4
                #outputs=outputs4
                preds=outputs.detach().cpu().numpy().squeeze(0)

        binary_map = (preds > thred).astype(np.uint8)[0]
        nid=i.stem
        bmap_img=Image.fromarray(binary_map,'L')
        bmap_img.save(output_path+nid+'.png')

if __name__ == "__main__":
    main()
