import os
import cv2
import imageio
import torch
import torch.nn.functional as F
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def change4save(img):
    img = img.detach().cpu().numpy()
    img = img.squeeze()
    img = img.transpose([1,2,0])
    img = (img*255.0).astype('uint8')
    return img

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

def main(args):

    inference_time = args.time_period # Denotes "t" in our paper
    i0_path = args.frame0
    i1_path = args.frame1
    img0 = cv2.imread(i0_path)[:, :, ::-1]
    img1 = cv2.imread(i1_path)[:, :, ::-1]

    img0 = (torch.from_numpy(img0.copy()).permute(2, 0, 1)/255.).unsqueeze(0).to(DEVICE)
    img1 = (torch.from_numpy(img1.copy()).permute(2, 0, 1)/255.).unsqueeze(0).to(DEVICE)

    n, c, h, w = img1.shape

    from core.models.VTinker_half import modelVFI 
    model = modelVFI()

    param = torch.load(args.model_file)
    tmp_param = model.state_dict()
    for kk,v in param.items():
        k = kk[7:]
        if k in tmp_param:
            tmp_param[k] = v
        else:
            print(k, "  Not load!!!")

    model.load_state_dict(tmp_param)
    model = model.half()
    model.eval().to(DEVICE)

    flow_deep, skip_num = get_deep_skip(img1)
    divisor = 2**(flow_deep+2)

    if (h % divisor != 0) or (w % divisor != 0):
        ph = ((h - 1) // divisor + 1) * divisor
        pw = ((w - 1) // divisor + 1) * divisor
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding, "constant", 0.5)
        img1 = F.pad(img1, padding, "constant", 0.5)

    with torch.no_grad():
        img = model.inference(img0.half(), img1.half(), inference_time, flow_deep, skip_num)
        img = torch.clamp(img.float(), 0, 1)
        imageio.imsave(args.save_dir, change4save(img[:, :, :h, :w]))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="interpolate for given pair of images")
    parser.add_argument("--frame0", type=str, required=True,
            help="file path of the first input frame")
    parser.add_argument("--frame1", type=str, required=True,
            help="file path of the second input frame")
    parser.add_argument("--time_period", type=float, default=0.5,
            help="time period for interpolated frame")
    
    parser.add_argument("--save_dir", type=str,
            default="./assets/results.jpg",
            help="dir to save interpolated frame")
    
    parser.add_argument('--model_file', type=str,
            default=r"./checkpoints/VTinker.pkl",
            help='weight of VTinker')
    
    args = parser.parse_args()

    main(args)
    # python -m tools.test_single_case --frame0 ./assets/img0.jpg --frame1 ./assets/img1.jpg --time_period 0.5 --save_dir ./assets/results.jpg --model_file ./checkpoints/VTinker.pkl