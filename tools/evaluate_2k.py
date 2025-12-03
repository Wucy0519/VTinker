import argparse
import os

import os.path as osp
import tempfile
import warnings

import cv2
import numpy as np
import torch
from tqdm import tqdm
import sys

sys.path.append("./tools")

from calc_metric import Metrics
from tools_test import IOBuffer, Tools
import torch.nn.functional as F

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

tmpDir = r"dataset_train_test/tmp/"
dataset_name = "xiph2k" # Choose from ["xiph2k", "xiph4k", "davis480p", "davis1080p"]

os.makedirs(osp.join(tmpDir, "disImages"), exist_ok=True)
os.makedirs(osp.join(tmpDir, "refImages"), exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


method_name = "VTinker"
SCALE = 2
input_frames = 2
print("Testing on Dataset: ", dataset_name)
print("Running VFI method : ", method_name)
print("TMP (temporary) Dir: ", tmpDir)

############ build Dataset ############
dataset = dataset_name

if dataset.lower() == "xiph2k":
    dstDir = r"dataset_train_test/Xiph/"
    RESCALE = "2K"
    videos = sorted(os.listdir(dstDir))
elif dataset.lower() == "xiph4k":
    dstDir = r"dataset_train_test/Xiph/"
    RESCALE = "4K"
    videos = sorted(os.listdir(dstDir))
elif dataset.lower() == "davis480p":
    dstDir = r"dataset_train_test/DAVIS_trainval/JPEGImages/480p/"
    RESCALE = None
    videos = sorted(os.listdir(dstDir))
elif dataset.lower() == "davis1080p":
    dstDir = r"dataset_train_test/DAVIS_trainval/JPEGImages/Full-Resolution/"
    RESCALE = "1080P"
    videos = sorted(os.listdir(dstDir))
elif dataset.lower() == "custom":
    dstDir = "data/custom"
    RESCALE = None
    videos = sorted(os.listdir(dstDir))
else:
    raise NotImplementedError("Unsupported Dataset")

metrics = ["psnr", "ssim", "lpips", "dists", "flolpips"] # Choose your metrics
print("Building SCORE models...", metrics)
metric = Metrics(metrics, skip_ref_frames=SCALE)
print("Done")

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


def get_deep_skip(img):
    n, c, h, w = img.shape
    lenss = max(h, w)/1024
    if lenss >= 3:
        flow_deep = 5
        skip_num = 1
    elif lenss >= 1.5:
        flow_deep = 4
        skip_num = 0
    else:
        flow_deep = 3
        skip_num = 0
    return flow_deep, skip_num


def inferRGB(*inputs):
    # divisor = 2**(8)
    inputs = [data.to(DEVICE, dtype=torch.float, non_blocking=True) for data in inputs] #[x.to(DEVICE) for x in inputs]
    outputs = []
    [img0, img1]  = inputs
    n, c, h, w = img1.shape
    flow_deep, skip_num = get_deep_skip(img1)
    divisor = 2**(flow_deep+2)

    if (h % divisor != 0) or (w % divisor != 0):
        ph = ((h - 1) // divisor + 1) * divisor
        pw = ((w - 1) // divisor + 1) * divisor
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding, "constant", 0.5)
        img1 = F.pad(img1, padding, "constant", 0.5)

    for time in range(SCALE - 1):
        t = (time + 1) / SCALE
        with torch.no_grad():
            img = model.inference(img0, img1, t, flownet_deep=flow_deep, skip_num=skip_num)
        tenOut = torch.clamp(img, 0, 1)
        tenOut = tenOut[:, :, :h, :w]
        outputs.append(tenOut.cpu())
    return outputs

############ load videos, interpolate, calc metric ############
print(len(videos))
scores = {}
for vid_name in tqdm(videos):
    sequences = [
        x for x in os.listdir(osp.join(dstDir, vid_name)) if ".jpg" in x or ".png" in x
    ]
    sequences.sort(key=lambda x: int(x[:-4]))
    sequences = [osp.join(dstDir, vid_name, x) for x in sequences]
    targetSeq = sequences[: (len(sequences) - 1) // SCALE * SCALE + 1]

    ############ write reference video frames ############
    out_dir = osp.join(tmpDir, "refImages")
    print(vid_name)
    for cnt, f in enumerate(targetSeq):
        frame = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        frame = Tools.resize(frame, RESCALE)
        cv2.imwrite(f"{out_dir}/{cnt:0>7d}.png", frame)
    height, width, _ = frame.shape
    tot_frames = len(targetSeq)
    print("VIDEO: ", vid_name, " (%d x %d x %d)" % (tot_frames, height, width))

    ############ build buffer with multi-threads ############
    inputSeq = Tools.sample_sequence(targetSeq, interval=SCALE)
    IO = IOBuffer(RESCALE, inp_num=input_frames)
    IO.start(inputSeq, osp.join(tmpDir, "disImages"))

    ############ interpolation & write distorted frames ############
    inps = IO.reader.get()  # e.g., [I1 I3]
    IO.writer.put(Tools.toArray(inps[0]))
    while True:
        outs = inferRGB(*inps)  # e.g., [I2]
        for o in Tools.toArray(outs + [inps[-input_frames // 2]]):
            IO.writer.put(o)
        inps = IO.reader.get()
        if inps is None:
            break
    IO.stop()

    disPth = osp.join(tmpDir, "dis.yuv")
    refPth = osp.join(tmpDir, "ref.yuv")
    Tools.frames2rawvideo(osp.join(tmpDir, "disImages", "*.png"), disPth)
    Tools.frames2rawvideo(osp.join(tmpDir, "refImages", "*.png"), refPth)

    ############ calc metric ############
    meta = dict(
        tmpDir=tmpDir,
        disImgs=osp.join(tmpDir, "disImages"),
        refImgs=osp.join(tmpDir, "refImages"),
        disMP4=disPth,
        refMP4=refPth,
        scale=SCALE,
        hwt=(height, width, tot_frames),
    )
    print("Calculating metrics")
    s = metric.eval(meta)
    scores[vid_name] = s
    print({k: f"{v:.4f}" for k, v in s.items()})

    os.system("rm -rf %s/*/*.png" % tmpDir)
    os.system("rm -rf %s/*.yuv" % tmpDir)


avg_score = {k: np.mean([v[k] for v in scores.values()]) for k in metrics}
print("AVG Score of %s".center(41, "=") % method_name)
for k, v in avg_score.items():
    print("{:<10} {:<10.3f}".format(k, v))

# python -m tools.evaluate_2k