import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from importlib import import_module
from torch.optim import AdamW
import torch.optim as optim
import itertools
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
# from .loss import EPE, Ternary, LapLoss

from core.models.VTinker import modelVFI as base_model
from core.models import Vgg19


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReconstructionLoss(nn.Module):
    def __init__(self, type='l1'):
        super(ReconstructionLoss, self).__init__()
        if (type == 'l1'):
            self.loss = nn.L1Loss()
        elif (type == 'l2'):
            self.loss = nn.MSELoss()
        else:
            raise SystemExit('Error: no such type of ReconstructionLoss!')

    def forward(self, sr, hr):
        return self.loss(sr, hr)
    
def vgg_loss(out, ref):
    weights = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5]
    final_loss = None
    for m, r, w in zip(out, ref, weights):
        if final_loss == None:
            final_loss = torch.mean(torch.abs(m - r))*w
        else:
            final_loss = final_loss + torch.mean(torch.abs(m - r))*w
    return final_loss

def style_loss(out, ref):
    weights = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5]
    final_loss = None
    for m, r, w in zip(out, ref, weights):
        B, C, H, W = m.shape
        m = m.view(B, C, H * W)
        m = torch.bmm(m, m.transpose(1, 2)) / (H * W)
        B, C, H, W = r.shape
        r = r.view(B, C, H * W)
        r = torch.bmm(r, r.transpose(1, 2)) / (H * W)

        if final_loss == None:
            final_loss = torch.mean((m - r)**2)*w
        else:
            final_loss = final_loss + torch.mean((m - r)**2)*w
    return final_loss

class Pipeline:
    def __init__(self,
            model_cfg_dict,
            optimizer_cfg_dict=None,
            local_rank=-1,
            training=False,
            resume=False,
            upr_redesgin_path = None
            ):
        self.model_cfg_dict = model_cfg_dict
        self.optimizer_cfg_dict = optimizer_cfg_dict
        self.upr_redesgin_path = upr_redesgin_path
        self.rec_loss = ReconstructionLoss(type="l1")
        self.vgg19 = Vgg19.Vgg19(requires_grad=False).eval()

        self.init_model()
        self.device()
        self.training = training
        
        # We note that in practical, the `lr` of AdamW is reset from the
        # outside, using cosine annealing during the while training process.
        if training:
            p_contral = []
            p_other = []
            for pname,p in self.model.named_parameters():
                if pname.startswith('flowgen'):
                    p_contral.append(p) 
                else:
                    p_other.append(p)
            self.optimG = AdamW([{'params':p_other},{'params':p_contral, 'lr':1.5e-6}],
                weight_decay=optimizer_cfg_dict["weight_decay"])

        # `local_rank == -1` is used for testing, which does not need DDP
        if local_rank != -1:
            self.model = DDP(self.model, device_ids=[local_rank],
                    output_device=local_rank, find_unused_parameters=True)

        # Restart the experiment from last saved model, by loading the state of
        # the optimizer
        if resume:
            assert training, "To restart the training, please init the"\
                    "pipeline with training mode!"
            print("Load optimizer state to restart the experiment")
            ckpt_dict = torch.load(optimizer_cfg_dict["ckpt_file"])
            self.optimG.load_state_dict(ckpt_dict["optimizer"])


    def train(self):
        self.model.train()


    def eval(self):
        self.model.eval()


    def device(self):
        self.model.to(DEVICE)
        self.vgg19.to(DEVICE)


    @staticmethod
    def convert_state_dict(rand_state_dict, pretrained_state_dict):
        param =  {
            k.replace("module.", "", 1): v
            for k, v in pretrained_state_dict.items()
            }
        param = {k: v
                for k, v in param.items()
                if ((k in rand_state_dict) and (rand_state_dict[k].shape \
                        == param[k].shape))
                }
        rand_state_dict.update(param)
        return rand_state_dict


    def init_model(self):

        def load_pretrained_state_dict(model, model_file):
            if (model_file == "") or (not os.path.exists(model_file)):
                raise ValueError(
                        "Please set the correct path for pretrained model!")

            print("Load pretrained model from %s."  % model_file)
            rand_state_dict = model.state_dict()
            pretrained_state_dict = torch.load(model_file)

            return Pipeline.convert_state_dict(
                    rand_state_dict, pretrained_state_dict)

        # check args
        model_cfg_dict = self.model_cfg_dict
        load_pretrain = model_cfg_dict["load_pretrain"] \
                if "load_pretrain" in model_cfg_dict else False
        model_file = model_cfg_dict["model_file"] \
                if "model_file" in model_cfg_dict else ""
        
        
        self.model = base_model()
        params = torch.load(self.upr_redesgin_path)
        tmp_param = self.model.flowgen.state_dict()

        for kk, v in params.items():
            k = kk[7:]
            if k in tmp_param:
                tmp_param[k] = v
        self.model.flowgen.load_state_dict(tmp_param)

        # load pretrained model weight
        if load_pretrain:
            state_dict = load_pretrained_state_dict(
                    self.model, model_file)
            self.model.load_state_dict(state_dict)
        else:
            print("Train from random initialization.")


    def save_optimizer_state(self, path, rank, step):
        if rank == 0:
            optimizer_ckpt = {
                     "optimizer": self.optimG.state_dict(),
                     "step": step
                     }
            torch.save(optimizer_ckpt, "{}/optimizer-ckpt.pth".format(path))


    def save_model(self, path, rank, save_step=None):
        if (rank == 0) and (save_step is None):
            torch.save(self.model.state_dict(), '{}/model.pkl'.format(path))
        if (rank == 0) and (save_step is not None):
            torch.save(self.model.state_dict(), '{}/model-{}.pkl'\
                    .format(path, save_step))

    def inference(self, img0, img1, time_period=0.5, flownet_deep = None, skip_num = 0):
        interp_img = self.model.module.inference(img0, img1,
                time_period=time_period,
                flownet_deep = flownet_deep, 
                skip_num = skip_num)
        return interp_img


    def train_one_iter(self, img0, img1, gt, jindu, learning_rate, time_period=0.5, flownet_deep = None, skip_num = 0):
        self.optimG.param_groups[0]['lr'] =learning_rate
        self.train()

        out = self.model(img0, img1, time_period, 
                         flownet_deep = flownet_deep, 
                         skip_num = skip_num)
        
        labels = [gt, img0, img1]
        vgg_loss_list = []
        rec_loss_list = []
        style_loss_list = []

        for imgt_out, imgt_ in zip(out, labels):
            loss_rec_ = 1.*self.rec_loss(imgt_out, imgt_)
            out_feat = self.vgg19((imgt_out + 1.) / 2.)
            
            with torch.no_grad():
                ref_feat = self.vgg19((imgt_.detach() + 1.) / 2.)
            rec_loss_list.append(loss_rec_)
            vgg_loss_list.append(vgg_loss(out_feat, ref_feat))
            style_loss_list.append(style_loss(out_feat, ref_feat))
        

        with torch.no_grad():
            loss_interp_l2_nograd = (((out[0] - gt) ** 2 + 1e-6) ** 0.5).mean()

        loss_vgg_ = (vgg_loss_list[0]*8. + vgg_loss_list[1] +vgg_loss_list[2])/10.
        loss_rec_ = (rec_loss_list[0]*8. + rec_loss_list[1] +rec_loss_list[2])/10.
        loss_style_ = (style_loss_list[0]*8. + style_loss_list[1] +style_loss_list[2])/10.
        if jindu < 0.5:
            loss_G = loss_rec_  + loss_vgg_
        else:
            loss_G = loss_rec_ + 0.25*loss_vgg_ + 40.0*loss_style_ #Follow FILM setting

        self.optimG.zero_grad()
        loss_G.backward()
        self.optimG.step()

        extra_dict = {}
        extra_dict["loss_interp_l2"] = loss_interp_l2_nograd

        return out[0], extra_dict



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pass
