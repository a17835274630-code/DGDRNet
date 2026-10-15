"""Report exact parameters and convolution-only MACs; FFT cost is not counted."""
import argparse
from dataclasses import asdict
import json
import torch
from torch import nn
from dgdrnet import DGDRNet,ModelConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--config',default='configs/wtb_cdd.json')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args();torch.set_num_threads(2)
    cfg=json.load(open(args.config,encoding='utf-8'))
    model=DGDRNet(ModelConfig(**cfg['model'])).to(args.device).eval()
    macs=[0]
    def hook(module,inputs,output):
        macs[0]+=output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
    handles=[m.register_forward_hook(hook) for m in model.modules() if isinstance(m,nn.Conv2d)]
    # PriorConv uses functional convolution and is counted separately.
    from dgdrnet.model import PriorConv
    def prior_hook(module,inputs,output):macs[0]+=output.numel()*module.weight.shape[-1]**2
    handles += [m.register_forward_hook(prior_hook) for m in model.modules() if isinstance(m,PriorConv)]
    with torch.no_grad():model(torch.zeros(1,3,args.size,args.size,device=args.device))
    for handle in handles:handle.remove()
    print(json.dumps({'parameters':sum(p.numel() for p in model.parameters()),'conv_only_macs':macs[0],
                      'note':'Not full FLOPs: excludes FFT, elementwise operations, pooling, nonlinearities and bias additions. Do not compare directly with paper FLOPs.'},indent=2))


if __name__=='__main__':main()
