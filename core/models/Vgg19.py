import torch
from torchvision import models
from ..models.utils import MeanShift


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False, rgb_range=1):
        super(Vgg19, self).__init__()
        
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        # self.slice1 = torch.nn.Sequential()
        # for x in range(30):
        #     self.slice1.add_module(str(x), vgg_pretrained_features[x])
        self.slice1 = vgg_pretrained_features[:4]
        self.slice2 = vgg_pretrained_features[4:9]
        self.slice3 = vgg_pretrained_features[9:14]
        self.slice4 = vgg_pretrained_features[14:23]
        self.slice5 = vgg_pretrained_features[23:]

        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = False
            for param in self.slice2.parameters():
                param.requires_grad = False
            for param in self.slice3.parameters():
                param.requires_grad = False
            for param in self.slice4.parameters():
                param.requires_grad = False
            for param in self.slice5.parameters():
                param.requires_grad = False
        
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

    def forward(self, X):
        h = self.sub_mean(X)
        out_list = []
        h = self.slice1(h)
        out_list.append(h)
        h = self.slice2(h)
        out_list.append(h)
        h = self.slice3(h)
        out_list.append(h)
        h = self.slice4(h)
        out_list.append(h)
        h = self.slice5(h)
        out_list.append(h)
        return out_list


if __name__ == '__main__':
    vgg19 = Vgg19(requires_grad=False)