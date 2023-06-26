import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
from tqdm import tqdm
from data.dataset import TSUNAMIDataset


def make_dataset(path=".\\TSUNAMI", out_path="dataset.pt"):
    dataset = TSUNAMIDataset()
    for i in tqdm(range(100)):
        file_t0 = path + "\\t0\\{:08d}.jpg".format(i)
        t0 = np.array(cv2.imread(file_t0)).T
        file_t1 = path + "\\t1\\{:08d}.jpg".format(i)
        t1 = np.array(cv2.imread(file_t1)).T
        t = np.concatenate((t0, t1), axis=1)

        file_mask = path + "\\mask\\{:08d}.png".format(i)
        out_1 = np.array(cv2.imread(file_mask, cv2.IMREAD_GRAYSCALE)).T // 255
        out_0 = np.ones_like(out_1) - out_1
        out = np.stack([out_0, out_1])

        x = torch.tensor(t).float()
        y = torch.tensor(out).float()
        dataset.add_data((x, y))
    torch.save(dataset, out_path)
