import os.path as osp
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from imageio import NibabelIO

BASE_PATH = "/home/data/MICCAI_FeTS2022_TrainingData"

class NibabelDataset(Dataset):
    def __init__(self,
                 base_path: str = BASE_PATH,
                 split: str = "train",
                 partitioning: str = "partitioning_1.csv",
                 partition_id: int = 0):
        self.base_path = base_path
        self.split = split
        self.data_df = pd.read_csv(osp.join(base_path, partitioning))
        self.data_df = self.data_df[self.data_df["Partition_ID"] == partition_id]

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        subject_id = self.data_df.iloc[idx, 0]
        img_path = osp.join(self.base_path, subject_id)
        img_flair_file = osp.join(img_path, "{}_flair.nii.gz".format(subject_id))
        img_t1_file = osp.join(img_path, "{}_t1.nii.gz".format(subject_id))
        img_t1ce_file = osp.join(img_path, "{}_t1ce.nii.gz".format(subject_id))
        img_t2_file = osp.join(img_path, "{}_t2.nii.gz".format(subject_id))
        seg_file = osp.join(img_path, "{}_seg.nii.gz".format(subject_id))
        nibio = NibabelIO()
        img_flair, dctimg_flair = nibio.read_images([img_flair_file])
        img_t1, dctimg_t1 = nibio.read_images([img_t1_file])
        img_t1ce, dctimg_t1ce = nibio.read_images([img_t1ce_file])
        img_t2, dctimg_t2 = nibio.read_images([img_t2_file])
        seg, dctseg = nibio.read_seg(seg_file)
        img = np.concatenate((img_flair, img_t1, img_t1ce, img_t2), axis=0)
        seg[seg == 4] = 3
        return {'data': img, 'seg': seg}


