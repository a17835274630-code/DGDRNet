"""Generate fixed degraded counterparts without resizing or overwriting inputs."""
import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import shutil
import cv2
import numpy as np
from dgdrnet.data import image_index
from dgdrnet.degradation import Parameters,per_image_rng,sample_parameters,degrade_bgr


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sharp-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True,help='Writes images/, optional labels/, and parameters.csv.')
    parser.add_argument('--labels-dir',type=Path)
    parser.add_argument('--seed',type=int,default=3407)
    parser.add_argument('--fixed',nargs=4,type=float,metavar=('L','ANGLE','SIGMA','Q'))
    parser.add_argument('--format',choices=['png','jpg'],default='png')
    args=parser.parse_args()
    source=args.sharp_dir.resolve();output=args.output_dir.resolve()
    if source==output or source in output.parents or output in source.parents:
        parser.error('Source and destination directories must not overlap.')
    if output.exists() and any(output.iterdir()):raise FileExistsError('Output directory is not empty.')
    files=image_index(source)
    if args.labels_dir:
        for key in files:
            if not (args.labels_dir/(key+'.txt')).is_file():raise FileNotFoundError(f'Missing label: {key}')
    if args.fixed:
        length,angle,sigma,quality=args.fixed
        if any(int(v)!=v for v in (length,sigma,quality)) or length<1 or sigma<0 or not 1<=quality<=100:
            parser.error('L, sigma and Q must be integers; L >= 1, sigma >= 0, 1 <= Q <= 100.')
        fixed=Parameters(int(length),angle,int(sigma),int(quality))
    else:fixed=None
    output.mkdir(parents=True,exist_ok=True)
    with (output/'parameters.csv').open('w',encoding='utf-8',newline='') as log:
        fields=['source_image','output_image','seed','blur_length','blur_angle','noise_sigma','jpeg_quality','kernel_support']
        writer=csv.DictWriter(log,fieldnames=fields);writer.writeheader()
        for key,path in files.items():
            rng=per_image_rng(args.seed,key)
            params=fixed or sample_parameters(rng)
            image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None:raise ValueError(f'Cannot decode image: {path.name}')
            decoded,encoded=degrade_bgr(image,params,rng)
            target=output/'images'/(key+'.'+args.format);target.parent.mkdir(parents=True,exist_ok=True)
            if args.format=='jpg':
                encoded.tofile(target)
            else:
                ok,png=cv2.imencode('.png',decoded)
                if not ok:raise RuntimeError('PNG encoding failed.')
                png.tofile(target)
            if args.labels_dir:
                label=output/'labels'/(key+'.txt');label.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(args.labels_dir/(key+'.txt'),label)
            writer.writerow({'source_image':path.relative_to(source).as_posix(),'output_image':target.relative_to(output).as_posix(),
                             'seed':args.seed,**asdict(params),'kernel_support':params.blur_length if params.blur_length%2 else params.blur_length+1})
    print(f'Generated {len(files)} degraded images; sampled parameters saved to parameters.csv.')


if __name__=='__main__':main()
