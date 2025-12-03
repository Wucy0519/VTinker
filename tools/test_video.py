import cv2
import time
import torch
from torch.nn import functional as F

import argparse

def change4save(img):
    img = img.detach().cpu().numpy()
    img = img.squeeze()
    img = img.transpose([1,2,0])
    img = (img*255.0).astype('uint8')
    return img
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):

    from core.models.VTinker_half import modelVFI 
    param = torch.load(args.model_file)
    model = modelVFI()
    tmp_param = model.state_dict()
    for kk,v in param.items():
        k = kk[7:]
        if k in tmp_param:
            tmp_param[k] = v
    model.load_state_dict(tmp_param)
    model.half()
    model.to(DEVICE)
    model.eval()

    mp4_path = args.video#r"./test_img_video/test_video.mp4"

    cap = cv2.VideoCapture(mp4_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(args.save_dir, fourcc, fps*args.inter_num, (w, h))

    _, last_frame = cap.read()
    divisor = 2**(flow_deep+2)

    inter_num = args.inter_num
    j = 0 
    while True:
        output_video.write(last_frame)
        j += 1
        ret, frame = cap.read()
        if not ret:
            output_video.write(last_frame)
            break

        last_frame_tensor = (torch.tensor(last_frame.transpose(2, 0, 1)).to(DEVICE) / 255.).unsqueeze(0)
        frame_tensor = (torch.tensor(frame.transpose(2, 0, 1)).to(DEVICE) / 255.).unsqueeze(0)

        if (h % divisor != 0) or (w % divisor != 0):
            ph = ((h - 1) // divisor + 1) * divisor
            pw = ((w - 1) // divisor + 1) * divisor
            padding = (0, pw - w, 0, ph - h)
            last_frame_tensor = F.pad(last_frame_tensor, padding, "constant", 0.5)
            frame_tensor = F.pad(frame_tensor, padding, "constant", 0.5)
        
        for inter_i in range(1, inter_num):
            with torch.no_grad():
                img = model.inference(last_frame_tensor.half(), frame_tensor.half(), inter_i/inter_num, flow_deep, skip_num)
            img = torch.clamp(img.float(), 0, 1)
            output_video.write(change4save(img[:, :, :h, :w]))
        last_frame = frame
    output_video.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="interpolate for given pair of images")
    parser.add_argument("--video", type=str, required=True,
            help="file path of the input video")
    parser.add_argument("--inter_num", type=int, default=2,
            help="time period for interpolated frame")
    
    parser.add_argument("--save_dir", type=str,
            default=r"./test_img_video/result_video.mp4",
            help="dir to save interpolated frame")
    parser.add_argument('--model_file', type=str,
            default=r"./checkpoints/VTinker.pkl",
            help='weight of VTinker')

    args = parser.parse_args()
    main(args)
    # python -m tools.test_video --video ./assets/video.mp4 --inter_num 2 --save_dir ./assets/result_video.mp4 --model_file ./checkpoints/VTinker.pkl