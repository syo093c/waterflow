import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from dataset import SARDataset
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import rasterio
import ipdb

def display_sar(sar_data,num=6):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for i, ax in enumerate(axs.flatten()):
        print(i)
        ipdb.set_trace()
        ax.imshow(sar_data[i], cmap='gray')
        ax.set_title(f'Channel {i + 1}')
        ax.axis('off')
    #plt.show()

data_transforms = {
    "train": A.Compose(
        [
            # A.Resize(1024, 1024),
            # A.Downscale(scale_min=0.5,scale_max=0.9,p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=[10, 50]),
                    A.GaussianBlur(),
                    A.MotionBlur(),
                ],
                p=0.4,
            ),
            #ToTensorV2(),
        ],
        p=1.0,
    ),
}

train_dataset=SARDataset(
    image_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/images/",
    label_root="/home/syo/work/2024_IEEE_GRSS/dataset/Track1/train/labels/",
    mode='train',
    val_size=1e-10,
    data_transforms=data_transforms['train']
    )

for i in train_dataset:
    display_sar(i['data'])
    break