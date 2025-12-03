import torch
import torch.nn as nn
import torch.nn.functional as F
is_half = False
if is_half:
    from ..models.softsplat import softsplat_half as softsplat
    from ..utils import correlation_half as correlation
else:
    from ..models.softsplat import softsplat
    from ..utils import correlation
from .helper import ResidualBlock, NonLocalBlock, DownSampleBlock, UpSampleBlock, GroupNorm, Swish
import math

def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)

def resize_nearest(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="nearest")

def warp(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    if is_half:
        grid = torch.cat([xx, yy], 1).to(img).half()
    else:
        grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    if is_half:
        output = F.grid_sample(input=img, grid=grid_.half(), mode='bilinear', padding_mode='border', align_corners=True)
    else:
        output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output

def warp_nearest(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    if is_half:
        grid = torch.cat([xx, yy], 1).to(img).half()
    else:
        grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    if is_half:
        output = F.grid_sample(input=img, grid=grid_.half(), mode='nearest', padding_mode='border', align_corners=True)
    else:
        output = F.grid_sample(input=img, grid=grid_, mode='nearest', padding_mode='border', align_corners=True)
    return output

def upsample_flow(flow, mask, scale = 2):
    scale = int(scale)
    N, _, H, W = flow.shape
    mask = mask.view(N, 1, 9, scale, scale, H, W)#mask channel:9*4=36
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(scale * flow, [3, 3], padding=1)
    up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, 2, scale * H, scale * W)

class FeatPyramid(nn.Module):
    def __init__(self):
        super(FeatPyramid, self).__init__()
        self.conv_stage0 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_stage1 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                    stride=2, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_stage2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                    stride=2, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))

    def forward(self, img):
        C0 = self.conv_stage0(img)
        C1 = self.conv_stage1(C0)
        C2 = self.conv_stage2(C1)
        return [C0, C1, C2]

class MotionEstimator(nn.Module):
    def __init__(self):
        super(MotionEstimator, self).__init__()
        self.conv_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=275, out_channels=160,
                    kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer2 = nn.Sequential(
                nn.Conv2d(in_channels=160, out_channels=128,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=112,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer4 = nn.Sequential(
                nn.Conv2d(in_channels=112, out_channels=96,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer5 = nn.Sequential(
                nn.Conv2d(in_channels=96, out_channels=64,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer6 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=2,
                    kernel_size=3, stride=1, padding=1))


    def forward(self, feat0, feat1, last_feat, last_flow):
        b, _, _,_ = last_feat.shape
        corr_fn=correlation.FunctionCorrelation
        feat0_warp = warp(feat1, last_flow[:b//2]*0.25)
        feat1_warp = warp(feat0, last_flow[b//2:]*0.25)

        volume0 = F.leaky_relu(
                input=corr_fn(tenFirst=feat0, tenSecond=feat0_warp),
                negative_slope=0.1, inplace=False)
        volume1 = F.leaky_relu(
                input=corr_fn(tenFirst=feat1, tenSecond=feat1_warp),
                negative_slope=0.1, inplace=False)
        
        input_feat0 = torch.cat([volume0, feat0, feat0_warp,last_feat[:b//2],last_flow[:b//2]], 1)
        input_feat1 = torch.cat([volume1, feat1, feat1_warp,last_feat[b//2:],last_flow[b//2:]], 1)
        input_feat = torch.cat([input_feat0,input_feat1],dim=0)
        feat = self.conv_layer1(input_feat)
        feat = self.conv_layer2(feat)
        feat = self.conv_layer3(feat)
        feat = self.conv_layer4(feat)
        feat = self.conv_layer5(feat)
        flow= self.conv_layer6(feat)
        return flow, feat
    
class FlowGen(nn.Module):
    def __init__(self):
        super(FlowGen, self).__init__()
        self.feat_pyramid = FeatPyramid()
        self.motion_estimator = MotionEstimator()

    def forward(self, x):
        return x
    
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.Elayer1 = nn.Sequential(ResidualBlock(128, 128),
                                     ResidualBlock(128, 128),
                                     DownSampleBlock(128))#/4
        self.Elayer2 = nn.Sequential(ResidualBlock(128, 256),
                                     ResidualBlock(256, 256),
                                     DownSampleBlock(256))#/8
        self.Elayer3 = nn.Sequential(ResidualBlock(256, 256),
                                     ResidualBlock(256, 256),
                                     DownSampleBlock(256))#/16
        self.Elayer4 = nn.Sequential(ResidualBlock(256, 512),
                                     ResidualBlock(512, 512))
        self.Efinal =  nn.Sequential(ResidualBlock(512, 512),
                                     ResidualBlock(512, 512),
                                     GroupNorm(512),
                                     Swish(),
                                     nn.Conv2d(512, 256, 3, 1, 1))
        
        self.Dhead = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1),
                                   ResidualBlock(512, 512),
                                   ResidualBlock(512, 512))
        
        self.Dlayer0 = nn.Sequential(ResidualBlock(512, 512),
                                     ResidualBlock(512, 512))#/16
        
        self.Dlayer1 = nn.Sequential(ResidualBlock(512+256, 256),
                                     ResidualBlock(256, 256),
                                     UpSampleBlock(256))#/8
        
        self.Dlayer2 = nn.Sequential(ResidualBlock(256+256, 256),
                                     ResidualBlock(256, 256),
                                     UpSampleBlock(256))#/4
        
        self.Dlayer3 = nn.Sequential(ResidualBlock(256+128, 128),
                                     ResidualBlock(128, 128),
                                     UpSampleBlock(128))#/2
        
        self.Dlayer4 = nn.Sequential(ResidualBlock(128+128, 128),
                                     ResidualBlock(128, 128),
                                     UpSampleBlock(128))#/1
        
        self.Dfinal = nn.Sequential(GroupNorm(128),
                                    Swish(),
                                    nn.Conv2d(128, 3, 3, 1, 1))
        
    def forward(self, x):
        out0 = self.Elayer1(x)#/4
        out1 = self.Elayer2(out0)#/8
        out2 = self.Elayer3(out1)#/16
        out = self.Elayer4(out2)#/16
        out = self.Efinal(out)#/16
        out = self.Dhead(out)#/16
        out = self.Dlayer0(out)#/16
        out = self.Dlayer1(torch.cat([out, out2], dim=1))#/8
        out = self.Dlayer2(torch.cat([out, out1], dim=1))#/4
        out = self.Dlayer3(torch.cat([out, out0], dim=1))#/2
        out = self.Dlayer4(torch.cat([out,    x], dim=1))#/1
        out = self.Dfinal(out)
        return out

class LiteUnet(nn.Module):
    def __init__(self):
        super(LiteUnet, self).__init__()
        self.Elayer1 = nn.Sequential(ResidualBlock(128, 32),
                                     DownSampleBlock(32))#/2
        self.Elayer2 = nn.Sequential(ResidualBlock(32, 64),
                                     DownSampleBlock(64))#/4
        self.Elayer3 = nn.Sequential(ResidualBlock(64, 128),
                                     DownSampleBlock(128))#/8
        self.Elayer4 = nn.Sequential(ResidualBlock(128, 128))#/8
        
        self.Dlayer0 = nn.Sequential(ResidualBlock(128, 64))#/8
        
        self.Dlayer1 = nn.Sequential(ResidualBlock(64+128, 64),
                                     UpSampleBlock(64))#/4
        
        self.Dlayer2 = nn.Sequential(ResidualBlock(64+64, 128),
                                     UpSampleBlock(128))#/2
        
    def forward(self, x):
        out0 = self.Elayer1(x)#/2
        out1 = self.Elayer2(out0)#/4
        out2 = self.Elayer3(out1)#/8
        out = self.Elayer4(out2)#/8
        out = self.Dlayer0(out)#/8
        out = self.Dlayer1(torch.cat([out, out2], dim=1))
        out = self.Dlayer2(torch.cat([out, out1], dim=1))
        return out

class TBVFI(nn.Module):
    def __init__(self):
        super(TBVFI, self).__init__()
        self.tx_h, self.tx_w = (2,2)

        self.search_local = 3
        self.pad_num = self.search_local//2
        self.pad = torch.nn.ZeroPad2d(self.search_local//2) 
        self.search_size_h, self.search_size_w = (2, 2)

        self.merge_feat = nn.Sequential(ResidualBlock(256, 256),
                                     ResidualBlock(256, 128),
                                     ResidualBlock(128, 128))
        
        self.Ehead_warp = nn.Sequential(nn.Conv2d(6, 128, 3, 1, 1),
                                   ResidualBlock(128, 128),
                                   DownSampleBlock(128))#/2
        
        self.Ehead_img = nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1),
                                   ResidualBlock(128, 128),
                                   DownSampleBlock(128))#/2
        
        self.mid_net = Unet()
        self.lite_net = LiteUnet()

    def bis_local(self, input, dim, index):
        index = index.unsqueeze(-1)
        return torch.gather(input, dim, index)
    
    def bis(self, input, dim, index):
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def warping(self, img0, img1, flow, t):
        flow_0t = flow[:, :2] 
        flow_1t = flow[:, 2:4]
        img0_ = softsplat.FunctionSoftsplat(
                    tenInput=img0, tenFlow=flow_0t,
                    tenMetric=None, strType='average')
        img1_ = softsplat.FunctionSoftsplat(
                    tenInput=img1, tenFlow=flow_1t,
                    tenMetric=None, strType='average')
        return img0_, img1_
    
    def forward_to_backward(self, flow):
        B, _, H, W = flow.shape
        flow = flow.contiguous()
        warped_grid = softsplat.FunctionSoftsplat(
                tenInput=flow, tenFlow=flow,
                tenMetric=None, strType='average')
        backward_flow = -1*warped_grid
        flowt0 = backward_flow[:B//2]
        flowt1 = backward_flow[B//2:]
        return flowt0, flowt1#, grid[B//2:]
    
    def texture_search_local(self, Q, K0, K1, V0, V1):
        K = torch.cat([K0, K1], dim = 1)
        V = torch.cat([V0, V1], dim = 1)
        
        b, c, h, w = Q.shape
        # bk, ck, hk, hw = K.shape
        kernel_size_h, kernel_size_w = (self.search_size_h, self.search_size_w)
        stride_size_h, stride_size_w = (self.search_size_h, self.search_size_w)
        kernel_num = kernel_size_h * kernel_size_w
        search_Q = F.unfold(Q, 
                            kernel_size=(kernel_size_h, kernel_size_w),
                            stride=(stride_size_h, stride_size_w))
        search_Q = F.normalize(search_Q, dim = 1)
        search_Q = search_Q.reshape([b, c, kernel_num, -1, 1]).mean(dim=2).permute(0, 2, 1, 3)#b, 4096, 256,1

        search_K = F.unfold(K, 
                            kernel_size=(kernel_size_h, kernel_size_w),
                            stride=(stride_size_h, stride_size_w))
        num_h, num_w = h//self.search_size_h, w//self.search_size_w
        search_K = search_K.reshape([b, 2, c*kernel_num, num_h, num_w])
        search_K = F.normalize(search_K, dim = 2)
        search_K = search_K.reshape([b, 2*c,kernel_num, num_h, num_w]).mean(dim=2)
        search_K = self.pad(search_K)
        search_K = F.unfold(search_K, kernel_size=self.search_local)#b, 81*512, 4096
        search_K = search_K.permute(0, 2, 1).reshape([b, -1, 2, c, self.search_local**2])
        search_K = search_K.permute(0, 1, 2, 4, 3).reshape([b, -1, 2*self.search_local**2, c])#b, 4096, 162, 256

        index_qk = torch.einsum('bikj, bijt->bikt', search_K, search_Q).squeeze(-1).permute(0, 2, 1)#1, 2n, n
        qk_star, qk_arg = torch.max(index_qk, dim=1)
        # print(qk_arg)#b, 4096
        ind_real = torch.arange((num_h+2*self.pad_num)*(num_w+2*self.pad_num)*2).to(Q.device)
        ind_real = ind_real.reshape([1, 2, num_h+2*self.pad_num, num_w+2*self.pad_num]).repeat(b, 1, 1, 1)#b, 2, h+8, w+8
        ind_real = F.unfold(ind_real.float(), kernel_size=self.search_local).permute(0, 2, 1)#b, 4096, 2*81
        ind_real = self.bis_local(ind_real, 2, qk_arg).squeeze(-1)
        # print(ind_real)

        b, c, H, W = V0.shape
        kernel_size_h, kernel_size_w = (self.search_size_h*2, self.search_size_w*2)
        stride_size_h, stride_size_w = (self.search_size_h*2, self.search_size_w*2)
        kernel_num = kernel_size_h*kernel_size_w
        search_V = F.unfold(V,
                            kernel_size=(kernel_size_h, kernel_size_w),
                            stride=(stride_size_h, stride_size_w))
        search_V = search_V.reshape([b, -1, num_h, num_w])
        search_V = self.pad(search_V)
        search_V = search_V.reshape([b, 2, c*kernel_num, -1]).permute(0, 2, 1, 3).reshape([b, c*kernel_num, -1])
        
        search_V = self.bis(search_V, 2, ind_real.long())
        result_qk = F.fold(search_V, 
                            output_size=(H, W),
                            kernel_size=(kernel_size_h, kernel_size_w),
                            stride=(stride_size_h, stride_size_w))
        return result_qk

    def get_texture(self, v0, v1, flow, q, k0, k1):
        flow = resize(flow, 0.5)/2.
        flow_cat = torch.cat([flow[:, :2], flow[:, 2:4]], dim=0)
        flowt0, flowt1 = self.forward_to_backward(flow_cat)
        f0 = resize_nearest(flowt0, 1/self.tx_h)*(1/self.tx_h)
        f1 = resize_nearest(flowt1, 1/self.tx_h)*(1/self.tx_h)

        b, c, h, w = v0.shape
        v0_ = F.unfold(v0, kernel_size=(self.tx_h, self.tx_w), stride=1)
        v0_ = v0_.reshape([b, -1, h-(self.tx_h-1), w-(self.tx_w-1)])

        v1_ = F.unfold(v1, kernel_size=(self.tx_h, self.tx_w), stride=1)
        v1_ = v1_.reshape([b, -1, h-(self.tx_h-1), w-(self.tx_w-1)])

        b_, c_, h_, w_ = v0_.shape
        
        v0_ = warp_nearest(v0_, f0).reshape([b_, c_, -1])
        v1_ = warp_nearest(v1_, f1).reshape([b_, c_, -1])
        k0 = warp_nearest(k0, f0)
        k1 = warp_nearest(k1, f1)

        v0_ = F.fold(v0_, output_size=(h, w),
                        kernel_size=(self.tx_h, self.tx_w),
                        stride=(self.tx_h, self.tx_w))
        v1_ = F.fold(v1_, output_size=(h, w),
                        kernel_size=(self.tx_h, self.tx_w),
                        stride=(self.tx_h, self.tx_w))
        return self.texture_search_local(q, k0, k1, v0_, v1_)

    def forward(self, img0, img1, flow):
        flow4x = flow
        flow_0t = flow4x[:, :2].contiguous()
        flow_1t = flow4x[:, 2:4].contiguous()
        warped_img0 = softsplat.FunctionSoftsplat(
                tenInput=img0, tenFlow=flow_0t,
                tenMetric=None, strType='average')
        warped_img1 = softsplat.FunctionSoftsplat(
                tenInput=img1, tenFlow=flow_1t,
                tenMetric=None, strType='average')
        warped_feature = self.Ehead_warp(torch.cat([warped_img0, warped_img1], dim=1))
        Q = self.lite_net(warped_feature)
        V0 = self.Ehead_img(img0)
        K0 = self.lite_net(V0)
        V1 = self.Ehead_img(img1)
        K1 = self.lite_net(V1)

        It_feat = self.get_texture(V0, V1, flow, Q, K0, K1)
        input_It_feat = torch.cat([It_feat, warped_feature], dim=1)#128+128
        It_feat = warped_feature + self.merge_feat(input_It_feat)

        out = self.mid_net(It_feat)
        out0 = self.mid_net(V0)
        out1 = self.mid_net(V1)

        return [out, out0, out1]
    
    def inference(self, img0, img1, flow):
        flow4x = flow
        flow_0t = flow4x[:, :2].contiguous()
        flow_1t = flow4x[:, 2:4].contiguous()
        warped_img0 = softsplat.FunctionSoftsplat(
                tenInput=img0, tenFlow=flow_0t,
                tenMetric=None, strType='average')
        warped_img1 = softsplat.FunctionSoftsplat(
                tenInput=img1, tenFlow=flow_1t,
                tenMetric=None, strType='average')
        warped_feature = self.Ehead_warp(torch.cat([warped_img0, warped_img1], dim=1))
        Q = self.lite_net(warped_feature)
        V0 = self.Ehead_img(img0)
        K0 = self.lite_net(V0)
        V1 = self.Ehead_img(img1)
        K1 = self.lite_net(V1)

        It_feat = self.get_texture(V0, V1, flow, Q, K0, K1)
        input_It_feat = torch.cat([It_feat, warped_feature], dim=1)
        It_feat = warped_feature + self.merge_feat(input_It_feat)

        out = self.mid_net(It_feat)

        return out
    
class Up_sample_module(nn.Module):
    def __init__(self, hid_channel = 32):
        super(Up_sample_module, self).__init__()
        self.hc = hid_channel
        
        self.A = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.hc, kernel_size=3,
                stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=self.hc, out_channels=self.hc, kernel_size=3,
                stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1))
        
        self.B = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=self.hc, kernel_size=3,
                stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=self.hc, out_channels=self.hc, kernel_size=3,
                stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1))
        
        self.pred = nn.Sequential(
            nn.Conv2d(in_channels=2*self.hc, out_channels=self.hc, kernel_size=3,
                stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=self.hc, out_channels=self.hc, kernel_size=3,
                stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1))
        
        self.final = nn.Conv2d(in_channels=self.hc, out_channels=2, kernel_size=1, stride= 1, padding=0)

    def flow_up(self, flow, img):
        a = self.A(img)
        b = self.B(flow)
        input_feat = torch.cat([a*b, b], dim = 1)
        new_flow = self.pred(input_feat)
        return self.final(new_flow)
        
    def forward(self, flow, img):
        last_flow_mid = flow + self.flow_up(flow, img)
        return last_flow_mid

        

class modelVFI(nn.Module):
    def __init__(self, qkv_mode = "l"):
        super(modelVFI, self).__init__()
        self.flowgen = FlowGen()
        self.main_model = TBVFI()
        self.flownet_deep = 3

        self.flow_base = None
        self.flow_large = None
        self.up_sample = Up_sample_module(32) # GFU
        
    def get_flow(self, img0, img1, time_period, flownet_deep = None, skip_num = 0):
        if flownet_deep != None:
            self.flownet_deep = flownet_deep
        N, C, H, W = img0.shape
        down_rate = 2**(self.flownet_deep+2)
        if is_half:
            last_flow = torch.zeros((2*N, 2, H//down_rate, W //down_rate)).to(img0.device).half()
            last_feat = torch.zeros((2*N, 64, H//down_rate, W//down_rate)).to(img0.device).half()
        else:
            last_flow = torch.zeros((2*N, 2, H//down_rate, W //down_rate)).to(img0.device)
            last_feat = torch.zeros((2*N, 64, H//down_rate, W//down_rate)).to(img0.device)
        for rate in list(range(self.flownet_deep+1))[::-1]:
            if rate != 0:
                img0_ = resize(img0, 1./(2**rate))
                img1_ = resize(img1, 1./(2**rate))
            else:
                img0_ = img0
                img1_ = img1
            f0 = self.flowgen.feat_pyramid(img0_)
            f1 = self.flowgen.feat_pyramid(img1_)
            last_flow, last_feat = self.flowgen.motion_estimator(f0[-1], f1[-1], last_feat, last_flow)

            last_flow = resize(last_flow, 2.)*2.
            input_img = resize(torch.cat([img0_, img1_], dim=0), 1./2)
            last_flow = self.up_sample(last_flow, input_img)
            last_feat = (resize(last_feat, 2.))*2.
            if rate == skip_num:
                self.flow_base = last_flow.clone()
                last_flow = resize(last_flow, 2**(skip_num+1))*(2**(skip_num+1))
                input_img = torch.cat([img0, img1], dim=0)
                last_flow = self.up_sample(last_flow, input_img)
                self.flow_large = last_flow.clone()
                break

        flow_0t = last_flow[:N] * time_period
        flow_1t = last_flow[N:] * (1 - time_period)
        return torch.cat([flow_0t, flow_1t], dim=1)/4.

    def forward(self, img0, img1, time_period, flownet_deep = None, skip_num = 0):
        flow = self.get_flow(img0, img1, time_period, flownet_deep, skip_num)
        out_list = self.main_model(img0, img1, flow)
        return out_list
    
    def inference(self, img0, img1, time_period, flownet_deep = None, skip_num = 0):
        with torch.no_grad():
            flow = self.get_flow(img0, img1, time_period, flownet_deep, skip_num)
            out = self.main_model.inference(img0, img1, flow)
        return out
