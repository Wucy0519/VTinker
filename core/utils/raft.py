import torch
import torch.nn as nn
import torch.nn.functional as F


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def bilinear_sampler(img, coords, mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), 
                            torch.arange(wd, device=device), 
                            indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


class SmallUpdateBlock(nn.Module):
    def __init__(self, cdim, hidden_dim, flow_dim, corr_dim, fc_dim,
                 corr_levels=4, radius=3, scale_factor=None):
        super(SmallUpdateBlock, self).__init__()
        cor_planes = corr_levels * (2 * radius + 1) **2
        self.scale_factor = scale_factor

        self.convc1 = nn.Conv2d(2 * cor_planes, corr_dim, 1, padding=0)
        self.convf1 = nn.Conv2d(4, flow_dim*2, 7, padding=3)
        self.convf2 = nn.Conv2d(flow_dim*2, flow_dim, 3, padding=1)
        self.conv = nn.Conv2d(corr_dim+flow_dim, fc_dim, 3, padding=1)

        self.gru = nn.Sequential(
            nn.Conv2d(fc_dim+4+cdim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

        self.feat_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, cdim, 3, padding=1),
        )

        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, 4, 3, padding=1),
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            
    def forward(self, net, flow, corr):
        net = resize(net, 1 / self.scale_factor
                      ) if self.scale_factor is not None else net
        cor = self.lrelu(self.convc1(corr))
        flo = self.lrelu(self.convf1(flow))
        flo = self.lrelu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        inp = self.lrelu(self.conv(cor_flo))
        inp = torch.cat([inp, flow, net], dim=1)

        out = self.gru(inp)
        delta_net = self.feat_head(out)
        delta_flow = self.flow_head(out)
        
        if self.scale_factor is not None:
            delta_net = resize(delta_net, scale_factor=self.scale_factor)
            delta_flow = self.scale_factor * resize(delta_flow, scale_factor=self.scale_factor)
        
        return delta_net, delta_flow


class BasicUpdateBlock(nn.Module):
    def __init__(self, cdim, hidden_dim, flow_dim, corr_dim, corr_dim2, 
                 fc_dim, corr_levels=4, radius=3, scale_factor=None, out_num=1):
        super(BasicUpdateBlock, self).__init__()
        cor_planes = corr_levels * (2 * radius + 1) **2

        self.scale_factor = scale_factor
        self.convc1 = nn.Conv2d(2 * cor_planes, corr_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(corr_dim, corr_dim2, 3, padding=1)
        self.convf1 = nn.Conv2d(4, flow_dim*2, 7, padding=3)
        self.convf2 = nn.Conv2d(flow_dim*2, flow_dim, 3, padding=1)
        self.conv = nn.Conv2d(flow_dim+corr_dim2, fc_dim, 3, padding=1)

        self.gru = nn.Sequential(
            nn.Conv2d(fc_dim+4+cdim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

        self.feat_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, cdim, 3, padding=1),
        )

        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, 4*out_num, 3, padding=1),
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            
    def forward(self, net, flow, corr):
        net = resize(net, 1 / self.scale_factor
                      ) if self.scale_factor is not None else net
        cor = self.lrelu(self.convc1(corr))
        cor = self.lrelu(self.convc2(cor))
        flo = self.lrelu(self.convf1(flow))
        flo = self.lrelu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        inp = self.lrelu(self.conv(cor_flo))
        inp = torch.cat([inp, flow, net], dim=1)

        out = self.gru(inp)
        delta_net = self.feat_head(out)
        delta_flow = self.flow_head(out)
        
        if self.scale_factor is not None:
            delta_net = resize(delta_net, scale_factor=self.scale_factor)
            delta_flow = self.scale_factor * resize(delta_flow, scale_factor=self.scale_factor)
        return delta_net, delta_flow


class BidirCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.corr_pyramid_T = []
        
        corr = BidirCorrBlock.corr(fmap1, fmap2)
        self.corr = corr
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr_T = corr.clone().permute(0, 4, 5, 3, 1, 2)

        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        corr_T = corr_T.reshape(batch*h2*w2, dim, h1, w1)
        
        self.grid = coords_grid(batch,h1,w1,self.corr.device)
        self.corr_pyramid.append(corr)
        self.corr_pyramid_T.append(corr_T)

        for _ in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            corr_T = F.avg_pool2d(corr_T, 2, stride=2)
            self.corr_pyramid.append(corr)
            self.corr_pyramid_T.append(corr_T)

    def GM_Flow(self):
        N, h1, w1, dim, h2, w2 = self.corr.shape
        corrMap = self.corr.view(N, h1*w1, h2*w2)
        prob0 = F.softmax(corrMap,dim=2) # (N, fH*fW)
        prob1 = F.softmax(corrMap,dim=1).permute(0,2,1).contiguous()
        prob = torch.cat([prob0,prob1],dim=0)
        ones = torch.ones(N*2,2,h1,w1).view(N*2, 2, -1).permute(0, 2, 1).to(self.corr.device)
        
        correspondence = torch.matmul(prob, ones).view(N*2, h1, w1, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]
        correspondence = correspondence - correspondence.detach() + torch.matmul(prob, self.grid).view(N*2, h1, w1, 2).permute(0, 3, 1, 2).detach()
        flow = correspondence - self.grid
        return flow[:N],flow[N:]

    def __call__(self, flow0, flow1):
        r = self.radius
        coords0 = self.grid + flow0
        coords1 = self.grid + flow1
        coords0 = coords0.permute(0, 2, 3, 1)
        coords1 = coords1.permute(0, 2, 3, 1)
        assert coords0.shape == coords1.shape, f"coords0 shape: [{coords0.shape}] is not equal to [{coords1.shape}]"
        batch, h1, w1, _ = coords0.shape

        out_pyramid = []
        out_pyramid_T = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            corr_T = self.corr_pyramid_T[i]

            dx = torch.linspace(-r, r, 2*r+1, device=coords0.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords0.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)

            centroid_lvl_0 = coords0.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            centroid_lvl_1 = coords1.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            coords_lvl_0 = centroid_lvl_0 + delta_lvl
            coords_lvl_1 = centroid_lvl_1 + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl_0)
            corr_T = bilinear_sampler(corr_T, coords_lvl_1)
            corr = corr.view(batch, h1, w1, -1)
            corr_T = corr_T.view(batch, h1, w1, -1)
            out_pyramid.append(corr)
            out_pyramid_T.append(corr_T)

        out = torch.cat(out_pyramid, dim=-1)
        out_T = torch.cat(out_pyramid_T, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float(), out_T.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())
    
class BidirCorrBlock_s:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.fmap1_pyramid = []
        self.fmap2_pyramid = []
        self.fmap1 = fmap1
        self.fmap2 = fmap2  
        # corr = BidirCorrBlock.corr(fmap1, fmap2)
        # self.corr = corr
        # batch, h1, w1, dim, h2, w2 = corr.shape
        # corr_T = corr.clone().permute(0, 4, 5, 3, 1, 2)

        # corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        # corr_T = corr_T.reshape(batch*h2*w2, dim, h1, w1)
        

        # self.corr_pyramid.append(corr)
        # self.corr_pyramid_T.append(corr_T)

        self.fmap1_pyramid.append(fmap1)
        self.fmap2_pyramid.append(fmap2)
        for _ in range(self.num_levels-1):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.fmap1_pyramid.append(fmap1)
            self.fmap2_pyramid.append(fmap2)

    def GM_Flow(self):
        fmap1 = self.fmap1
        fmap2 = self.fmap2
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corrMap = torch.matmul(fmap1.transpose(1,2), fmap2)
        corrMap = corrMap.view(batch, ht, wd, 1, ht, wd)
        corrMap = corrMap / torch.sqrt(torch.tensor(dim).float())
        N, h1, w1, dim, h2, w2 = corrMap.shape
        corrMap = corrMap.view(N, h1*w1, h2*w2)
        match12, match_idx12 = corrMap.max(dim=2) # (N, fH*fW)
        match21, match_idx21 = corrMap.max(dim=1)

        # matched coords
        match12 = match_idx12.reshape(N, h1, w1)
        match12_x = match12 % w1
        match12_y = match12 // w1

        match21 = match_idx21.reshape(N, h1, w1)
        match21_x = match21 % w1
        match21_y = match21 // w1

        match12_xy = torch.stack([match12_x, match12_y], dim=1).float()
        match21_xy = torch.stack([match21_x, match21_y], dim=1).float()
        
        return match12_xy,match21_xy

    def __call__(self, coords, flow0, flow1):
        _,dim,_,_ = self.fmap1.shape
        r = self.radius
        k = 2*r + 1
        coords0 = coords + flow0
        coords0_ = coords - flow0
        coords1 = coords + flow1
        coords1_ = coords - flow1
        coords0 = coords0.permute(0, 2, 3, 1)
        coords1 = coords1.permute(0, 2, 3, 1)
        coords0_= coords0_.permute(0, 2, 3, 1)
        coords1_ = coords1_.permute(0, 2, 3, 1)
        assert coords0.shape == coords1.shape, f"coords0 shape: [{coords0.shape}] is not equal to [{coords1.shape}]"
        batch, h1, w1, _ = coords0.shape

        out_pyramid = []
        out_pyramid_T = []
        for i in range(self.num_levels):
            f1 = self.fmap1_pyramid[i]
            f2 = self.fmap2_pyramid[i]
            _,c,h,w = f1.shape
            dx = torch.linspace(-r, r, 2*r+1, device=coords0.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords0.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)
            delta_lvl = delta.reshape(1,k*k, 1, 1, 2).expand(batch,-1,h1,w1,2)

            centroid_lvl_0 = coords0.reshape(batch,1,h1,w1, 2) / 2**i
            centroid_lvl_1 = coords1.reshape(batch,1,h1,w1, 2) / 2**i
            centroid_lvl_0_ = coords0_.reshape(batch,1,h1,w1, 2) / 2**i
            centroid_lvl_1_ = coords1_.reshape(batch,1,h1,w1, 2) / 2**i

            coords_lvl_0 = (centroid_lvl_0 + delta_lvl).view(-1,h1,w1,2)
            coords_lvl_0_ = (centroid_lvl_0_ - delta_lvl).view(-1,h1,w1,2)
            coords_lvl_1 = (centroid_lvl_1 + delta_lvl).view(-1,h1,w1,2)
            coords_lvl_1_ = (centroid_lvl_1_ - delta_lvl).view(-1,h1,w1,2)

            f1 = f1.view(batch,1,c,h,w).expand(-1,k*k,-1,-1,-1).reshape(-1,c,h,w)
            f2 = f2.view(batch,1,c,h,w).expand(-1,k*k,-1,-1,-1).reshape(-1,c,h,w)

            coor1 = bilinear_sampler(f1, coords_lvl_0)
            coor1_ = bilinear_sampler(f2, coords_lvl_0_)
            fout1  = (coor1 * coor1_).sum(dim = 1).view(batch,k*k,h1,w1)
            
            coor2 = bilinear_sampler(f2, coords_lvl_1)
            coor2_ = bilinear_sampler(f1, coords_lvl_1_)
            fout2  = (coor2 * coor2_).sum(dim = 1).view(batch,k*k,h1,w1)

            out_pyramid.append(fout1)
            out_pyramid_T.append(fout2)

        out = torch.cat(out_pyramid, dim=1)
        out_T = torch.cat(out_pyramid_T, dim=1)
        return out,out_T

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())    