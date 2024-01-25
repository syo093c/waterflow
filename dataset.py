import torch
from torch.utils.data import Dataset,DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
import ipdb
import rasterio
from PIL import Image
import numpy as np
from tqdm.auto import tqdm


def min_max_normalize_per_channel(data):
    min_vals = np.min(data, axis=(1, 2), keepdims=True)  # 计算每个通道的最小值
    max_vals = np.max(data, axis=(1, 2), keepdims=True)  # 计算每个通道的最大值

    # 添加容错处理，如果最小值和最大值相等（即所有值都是0），则将最大值设置为1
    max_vals[max_vals == min_vals] = 1

    normalized_data = (data - min_vals) / (max_vals - min_vals)
    ipdb.set_trace()
    return normalized_data

class SARDataset(Dataset):
    def __init__(self,image_root,label_root,data_transforms=None,mode='train',val_size=0.2,seed=42):
        self.mode=mode
        images_list = sorted(list(glob(image_root + "*")))
        label_list = sorted(list(glob(label_root + "*")))
        self.transforms=data_transforms

        images_train, images_val, labels_train, labels_val = train_test_split(
            images_list,
            label_list,
            test_size=val_size,
            #stratify=label_list,
            random_state=seed,
        )
        self.images_train =   sorted(images_train)
        self.images_val   =   sorted(images_val)
        self.labels_train =   sorted(labels_train)
        self.labels_val = sorted(labels_val)

    def __len__(self):
        if self.mode=='train':
            return len(self.images_train)
        elif self.mode=='val':
            return len(self.images_val)

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
        if self.mode=='train':
            fpath=self.images_train[index]
            lpath=self.labels_train[index]
        if self.mode=='val':
            fpath=self.images_val[index]
            lpath=self.labels_val[index]

        # sar_data = rasterio.open(fpath,dtype=rasterio.uint8)
        sar_data = rasterio.open(fpath)
        label_map = np.array(Image.open(lpath))

        sar_data=sar_data.read()
        # normalization
        sar_data=self._sar_normalization(sar_data)

        # sar_data[sar_data<0]=0
        # sar_data=min_max_normalize_per_channel(sar_data)

        # data augmentation
        if self.transforms:
            sar_data=np.transpose(sar_data, (1, 2, 0))
            transformed_data = self.transforms(image=sar_data,mask=label_map)

            sar_data=transformed_data['image']
            sar_data=np.transpose(sar_data, (2, 0, 1))
            label_map=transformed_data['mask']

        # normalization
        #sar_data=self._sar_normalization(sar_data)

        data = {
            "data": torch.tensor(sar_data, dtype=torch.float32),
            "label": torch.tensor(label_map, dtype=torch.long),
        }
        return data

def build_dataloader(batch_size=4,num_workers=2,val_size=0.2,seed=42,data_transforms=None):
    DATASET_ROOT='/home/syo/work/2024_IEEE_GRSS/dataset/'
    TRACK1_ROOT='/home/syo/work/2024_IEEE_GRSS/dataset/Track1/'
    TRACK2_ROOT='/home/syo/work/2024_IEEE_GRSS/dataset/Track2/'    

    train_dataset=SARDataset(
        image_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/images/",
        label_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/labels/",
        mode='train',
        val_size=val_size,
        seed=seed,
        data_transforms=data_transforms
    )
    val_dataset=SARDataset(
        image_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/images/",
        label_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/labels/",
        mode='val',
        val_size=val_size,
        seed=seed,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=num_workers, shuffle=True, pin_memory=True)
    return train_dataloader,val_dataloader

def main():
    train_dataset=SARDataset(
        image_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/images/",
        label_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/labels/",
        mode='train',
        val_size=1e-10
    )
    val_dataset=SARDataset(
        image_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/images/",
        label_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/labels/",
        mode='val',
        val_size=1e-10
    )
    for i in tqdm(train_dataset):
        print(i)
    for i in tqdm(val_dataset):
        pass

if __name__ == '__main__':
    main()
