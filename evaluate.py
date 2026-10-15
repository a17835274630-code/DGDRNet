"""Evaluate a selected checkpoint on paired images without training."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from dgdrnet.data import PairedDataset
from dgdrnet.metrics import psnr,ssim
from dgdrnet.runtime import load_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--blur-dir',type=Path,required=True)
    parser.add_argument('--sharp-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--save-images',action='store_true')
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--cpu-threads',type=int,default=4)
    args=parser.parse_args();torch.set_num_threads(args.cpu_threads)
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Output directory is not empty.')
    dataset=PairedDataset(args.blur_dir,args.sharp_dir,size=args.size)
    model=load_model(args.checkpoint,args.device)
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    with torch.no_grad():
        for batch in DataLoader(dataset,batch_size=1,shuffle=False):
            pred=model(batch['blur'].to(args.device)).clamp(0,1)
            target=batch['sharp'].to(args.device)
            name=batch['name'][0]
            rows.append({'name':name,'psnr':psnr(pred,target).item(),'ssim':ssim(pred,target).item()})
            if args.save_images:
                path=args.output/'images'/(name+'.png');path.parent.mkdir(parents=True,exist_ok=True)
                array=(pred[0].cpu().permute(1,2,0).numpy()*255).round().astype(np.uint8)
                Image.fromarray(array).save(path)
    with (args.output/'per_image_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['name','psnr','ssim']);writer.writeheader();writer.writerows(rows)
    result={'images':len(rows),'psnr':sum(x['psnr'] for x in rows)/len(rows),'ssim':sum(x['ssim'] for x in rows)/len(rows)}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
