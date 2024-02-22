import torch
from torch.utils.data import Dataset,DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
import ipdb
import rasterio
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import albumentations as A
from sklearn.model_selection import KFold 

class SARDataset(Dataset):
    def __init__(self,data,targets=None,data_transforms=None):
        self.data=data
        self.targets=targets
        self.transforms=data_transforms

    def __len__(self):
            return len(self.data)

    def _sar_normalization(self,sar_data):
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

    def __getitem__(self, index):
        fpath=self.data[index]
        lpath=self.targets[index]

        # sar_data = rasterio.open(fpath,dtype=rasterio.uint8)
        sar_data = rasterio.open(fpath)
        label_map = np.array(Image.open(lpath))

        sar_data=sar_data.read()
        # normalization
        sar_data=self._sar_normalization(sar_data)

        # data augmentation
        if self.transforms:
            sar_data=np.transpose(sar_data, (1, 2, 0))
            transformed_data = self.transforms(image=sar_data,mask=label_map)
            sar_data=transformed_data['image']
            sar_data=np.transpose(sar_data, (2, 0, 1))
            label_map=transformed_data['mask']

        data = {
            # c, w, h
            "data": torch.tensor(sar_data, dtype=torch.float32),
            "label": torch.tensor(label_map, dtype=torch.float32).unsqueeze(0),
        }
        return data

def build_dataloader(batch_size=4,num_workers=2,val_size=0.2,seed=42,data_transforms=None):
    DATASET_ROOT='/home/syo/work/2024_IEEE_GRSS/dataset/'
    TRACK1_ROOT='/home/syo/work/2024_IEEE_GRSS/dataset/Track1/'
    TRACK2_ROOT='/home/syo/work/2024_IEEE_GRSS/dataset/Track2/'    
    images_list = sorted(list(glob(TRACK1_ROOT+'train/images/' + "*")))
    label_list = sorted(list(glob(TRACK1_ROOT+'train/labels/' + "*")))
    images_train, images_val, labels_train, labels_val = train_test_split(
        images_list,
        label_list,
        test_size=val_size,
        random_state=seed,
    )

    train_dataset=SARDataset(
        data=images_train,
        targets=labels_train,
        data_transforms=data_transforms
    )
    val_dataset=SARDataset(
        data=images_val,
        targets=labels_val,
        data_transforms=None
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=True)
    return train_dataloader,val_dataloader

def build_dataloader_w_pseudo(batch_size=4,num_workers=2,val_size=0.2,seed=42,data_transforms=None):
    DATASET_ROOT='/home/syo/work/2024_IEEE_GRSS/dataset/'
    TRACK1_ROOT='/home/syo/work/2024_IEEE_GRSS/dataset/Track1_w_pseudo/'
    TRACK2_ROOT='/home/syo/work/2024_IEEE_GRSS/dataset/Track2_w_pseudo/'    

    train_dataset=SARDataset(
        image_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1_w_pseudo/train/images/",
        label_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1_w_pseudo/train/labels/",
        mode='train',
        val_size=val_size,
        seed=seed,
        data_transforms=data_transforms
    )
    val_dataset=SARDataset(
        image_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1_w_pseudo/train/images/",
        label_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1_w_pseudo/train/labels/",
        mode='val',
        val_size=val_size,
        seed=seed,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=True)
    return train_dataloader,val_dataloader

def test_kflod():
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
        for d in tqdm(train_dataset):
            pass
        print(i)
        #train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
        #val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=True)

if __name__ == '__main__':
    test_kflod()
