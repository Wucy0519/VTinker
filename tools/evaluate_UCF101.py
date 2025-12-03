import os
import sys
sys.path.append("./tools")
import argparse
import math
import cv2
import numpy as np
from PIL import Image
import imageio
import torch
import torch.nn.functional as F
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader
import glob
import lpips
from datasets import UCF101
lpips_model = lpips.LPIPS(net="alex").to(DEVICE)

import tools_test
from calc_metric import Metrics
import torchvision.transforms.functional as TF
import os.path as osp

from core.models.VTinker import modelVFI 
model = modelVFI()
tmp_param = model.state_dict()

param = torch.load("checkpoints/VTinker.pkl")

for kk,v in param.items():
    k = kk[7:]
    if k in tmp_param:
        tmp_param[k] = v

model.load_state_dict(tmp_param)
model.to(DEVICE)
model.eval()

divisor = 2**(5)

data_root = r'dataset_train_test/ucf101_interp_ours/'
dataset_val = UCF101(data_root)
dataloader_val = DataLoader(dataset_val, batch_size=1)

# metrics = ["psnr", "ssim", "lpips"]
metrics = ["psnr", "ssim", "lpips", "ie", "dists"]
print("Building SCORE models...", metrics)
metric = Metrics(metrics, skip_ref_frames=None, batch_size=1)
print("Done")

tmpDir = r"dataset_train_test/tmp/"
os.makedirs(osp.join(tmpDir, "disImages"), exist_ok=True)
os.makedirs(osp.join(tmpDir, "refImages"), exist_ok=True)

for i, data in enumerate(dataloader_val):
    data = data.to(DEVICE, dtype=torch.float, non_blocking=True) / 255.
            
    img0 = data[:, :3]
    img1 = data[:, 3:6]
    gt = data[:, 6:]

    n, c, h, w = img1.shape

    if (h % divisor != 0) or (w % divisor != 0):
        ph = ((h - 1) // divisor + 1) * divisor
        pw = ((w - 1) // divisor + 1) * divisor
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding, "constant", 0.5)
        img1 = F.pad(img1, padding, "constant", 0.5)

    with torch.no_grad():
        img = model.inference(img0, img1, 0.5, flownet_deep=3, skip_num=0)

        img = torch.clamp(img, 0, 1)
    img, gt = img[:, :, :h, :w], gt
    disPth = f"{tmpDir}/disImages/{i:08d}.png"
    refPth = f"{tmpDir}/refImages/{i:08d}.png"
    TF.to_pil_image(img[0]).save(disPth)
    TF.to_pil_image(gt[0]).save(refPth)

meta = dict(disImgs=f"{tmpDir}/disImages", refImgs=f"{tmpDir}/refImages")
avg_score = metric.eval(meta)

print("AVG Score of %s".center(41, "=") % "VTinker")
for k, v in avg_score.items():
    print("{:<10} {:<10.3f}".format(k, v))

# python -m tools.evaluate_UCF101