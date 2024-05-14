import os, glob, random
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
from .data_utils import pkload


class LPBADataset(Dataset):
    def __init__(self, paths, atlas_path, transforms):
        self.paths = paths
        self.atlas_path = atlas_path
        self.transforms = transforms
    
    def __getitem__(self, index):
        path = self.paths[index]
        x = sitk.GetArrayFromImage(sitk.ReadImage(path))[None, ...]
        y = sitk.GetArrayFromImage(sitk.ReadImage(self.atlas_path))[None, ...]
        x = np.ascontiguousarray(x) # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x = torch.from_numpy(x).cuda().float()
        y = torch.from_numpy(y).cuda().float()
        return x, y

    def __len__(self):
        return len(self.paths)


class LPBAInferDataset(Dataset):
    def __init__(self, paths, atlas_path, label_dir, transforms):
        self.paths = paths
        self.atlas_path = atlas_path
        self.label_dir = label_dir
        self.transforms = transforms
    
    def __getitem__(self, index):
        path = self.paths[index]
        name = os.path.split(path)[1]
        x = sitk.GetArrayFromImage(sitk.ReadImage(path))[None, ...]
        y = sitk.GetArrayFromImage(sitk.ReadImage(self.atlas_path))[None, ...]
        x = torch.from_numpy(x).cuda().float()
        y = torch.from_numpy(y).cuda().float()
        x_seg = sitk.GetArrayFromImage(sitk.ReadImage(glob.glob(os.path.join(self.label_dir, name[:3] + "*"))[0]))[None, ...]
        x_seg = torch.from_numpy(x_seg).cuda().float()
        y_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.label_dir, "S01.delineation.structure.label.nii.gz")))

        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)
    