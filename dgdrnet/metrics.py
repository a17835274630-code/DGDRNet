"""Per-image RGB PSNR and Gaussian-window SSIM."""
import torch
from torch.nn import functional as F


def psnr(pred, target):
    mse = (pred.clamp(0,1)-target.clamp(0,1)).square().flatten(1).mean(1)
    return -10 * mse.clamp_min(1e-12).log10()


def ssim(pred, target):
    pred, target = pred.clamp(0,1), target.clamp(0,1)
    axis = torch.arange(11,device=pred.device,dtype=pred.dtype)-5
    gaussian = torch.exp(-axis.square()/(2*1.5**2))
    gaussian = gaussian/gaussian.sum()
    window = (gaussian[:,None]*gaussian[None,:])[None,None].repeat(pred.shape[1],1,1,1)
    def smooth(x):
        return F.conv2d(x,window,padding=5,groups=pred.shape[1])
    mx,my = smooth(pred),smooth(target)
    vx,vy = smooth(pred.square())-mx.square(),smooth(target.square())-my.square()
    covariance = smooth(pred*target)-mx*my
    return (((2*mx*my+0.01**2)*(2*covariance+0.03**2))/((mx.square()+my.square()+0.01**2)*(vx+vy+0.03**2))).flatten(1).mean(1)
