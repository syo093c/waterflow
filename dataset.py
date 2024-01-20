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
    def __init__(self,image_root,label_root,mode='train',val_size=0.2):
        self.mode=mode
        images_list = sorted(list(glob(image_root + "*")))
        label_list = sorted(list(glob(label_root + "*")))

        images_train, images_val, labels_train, labels_val = train_test_split(
            images_list,
            label_list,
            test_size=val_size,
            #stratify=label_list,
            random_state=42,
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
        #sar_data[sar_data<0]=0
        #sar_data=min_max_normalize_per_channel(sar_data)

        data= {
            'data': torch.tensor(sar_data,dtype=torch.float32),
            'label':torch.tensor(label_map,dtype=torch.long),
        }
        return data

def build_dataloader(batch_size=4,num_workers=2):
    DATASET_ROOT='/home/syo/work/2024_IEEE_GRSS/dataset/'
    TRACK1_ROOT='/home/syo/work/2024_IEEE_GRSS/dataset/Track1/'
    TRACK2_ROOT='/home/syo/work/2024_IEEE_GRSS/dataset/Track2/'    

    train_dataset=SARDataset(
        image_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/images/",
        label_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/labels/",
        mode='train'
    )
    val_dataset=SARDataset(
        image_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/images/",
        label_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/labels/",
        mode='val'
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        num_workers=num_workers, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1,
                        num_workers=num_workers, shuffle=True, pin_memory=True)
    return train_dataloader,val_dataloader

def main():
    train_dataset=SARDataset(
        image_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/images/",
        label_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/labels/",
        mode='train'
    )
    val_dataset=SARDataset(
        image_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/images/",
        label_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/labels/",
        mode='val'
    )
    for i in tqdm(train_dataset):
        ipdb.set_trace()
        print(i)
    for i in tqdm(val_dataset):
        pass

if __name__ == '__main__':
    main()
