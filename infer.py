"""Restore RGB images with no ground truth or annotation requirement."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from dgdrnet.data import image_index,load_rgb
from dgdrnet.runtime import load_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--cpu-threads',type=int,default=4)
    args=parser.parse_args();torch.set_num_threads(args.cpu_threads)
    if args.size<8 or args.size%8:parser.error('Size must be a positive multiple of eight.')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Output directory is not empty.')
    images=image_index(args.input);model=load_model(args.checkpoint,args.device)
    args.output.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for name,path in images.items():
            pred=model(load_rgb(path,args.size).unsqueeze(0).to(args.device)).clamp(0,1)
            target=args.output/(name+'.png');target.parent.mkdir(parents=True,exist_ok=True)
            array=(pred[0].cpu().permute(1,2,0).numpy()*255).round().astype(np.uint8)
            Image.fromarray(array).save(target)
    print(f'Restored {len(images)} images as lossless PNG files.')


if __name__=='__main__':main()
