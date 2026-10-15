"""Train fixed paired images; select checkpoints using validation PSNR only."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from dgdrnet import DGDRNet, ModelConfig, RestorationLoss
from dgdrnet.data import PairedDataset
from dgdrnet.metrics import psnr, ssim
from dgdrnet.runtime import FORMAT, seed_all, save_checkpoint, load_checkpoint


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    p,s,n = 0.0,0.0,0
    for batch in loader:
        target = batch['sharp'].to(device)
        pred = model(batch['blur'].to(device))
        p += psnr(pred,target).sum().item()
        s += ssim(pred,target).sum().item()
        n += target.shape[0]
    return {'psnr':p/n, 'ssim':s/n}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/wtb_cdd.json')
    parser.add_argument('--data-root',type=Path,default=Path('data/WTB-CDD'))
    parser.add_argument('--output',type=Path,default=Path('runs/wtb_cdd'))
    parser.add_argument('--train-blur',type=Path)
    parser.add_argument('--train-sharp',type=Path)
    parser.add_argument('--train-labels',type=Path)
    parser.add_argument('--val-blur',type=Path)
    parser.add_argument('--val-sharp',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--stop-after-epoch',type=int,help='Stop early for testing without changing the full cosine schedule.')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--cpu-threads',type=int,default=4)
    args=parser.parse_args()
    if args.workers<0 or args.cpu_threads<1:
        parser.error('Workers must be nonnegative; CPU threads must be positive.')
    torch.set_num_threads(args.cpu_threads)
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.epochs is not None: cfg['epochs']=args.epochs
    if cfg['epochs']<1 or cfg['batch_size']<1 or cfg['reference_batches']<1:
        parser.error('Epochs, batch size and reference batches must be positive.')
    args.output.mkdir(parents=True,exist_ok=True)
    resume=load_checkpoint(args.resume,'cpu') if args.resume else None
    if not resume and any(args.output.iterdir()):
        raise FileExistsError('Output directory is not empty. Choose a new directory or explicitly resume.')
    seed_all(cfg['seed'])
    device=torch.device(args.device)
    model=DGDRNet(ModelConfig(**cfg['model'])).to(device)
    criterion=RestorationLoss(**cfg['loss']).to(device)
    labels=args.train_labels or args.data_root/'Train'/'Sharp'/'labels'
    if not cfg['use_annotations']:
        if args.train_labels:
            parser.error('This configuration disables defect annotations.')
        labels=None
    training=PairedDataset(args.train_blur or args.data_root/'Train'/'Blur'/'images',
                           args.train_sharp or args.data_root/'Train'/'Sharp'/'images',labels,cfg['image_size'])
    validation=PairedDataset(args.val_blur or args.data_root/'VAL'/'Blur'/'images',
                             args.val_sharp or args.data_root/'VAL'/'Sharp'/'images',size=cfg['image_size'])
    generator=torch.Generator().manual_seed(cfg['seed'])
    loader=DataLoader(training,batch_size=cfg['batch_size'],shuffle=True,num_workers=args.workers,generator=generator)
    val_loader=DataLoader(validation,batch_size=1,shuffle=False,num_workers=args.workers)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'],eta_min=cfg['minimum_learning_rate'])
    start,best=0,float('-inf')
    if resume:
        # Epoch scheduling and architecture must not silently change on resume.
        if resume['config'] != cfg:
            raise ValueError('Resume requires the same configuration, including total epochs.')
        model.load_state_dict(resume['model'],strict=True)
        criterion.load_state_dict(resume['loss'])
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        start,best=resume['epoch'],resume['best_psnr']
        generator.set_state(resume['loader_rng'])
        torch.set_rng_state(resume['torch_rng'].cpu())
        random.setstate(resume['python_rng'])
        np.random.set_state(resume['numpy_rng'])
        if device.type=='cuda' and resume.get('cuda_rng'):
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
    elif criterion.weighted_pixel:
        # Deterministic, preselected batches, before any optimizer update.
        reference_loader=DataLoader(training,batch_size=cfg['batch_size'],shuffle=False,num_workers=0)
        model.eval()
        reference=[]
        with torch.no_grad():
            for i,batch in enumerate(reference_loader):
                if i>=cfg['reference_batches']:break
                pred=model(batch['blur'].to(device))
                reference.append(criterion.charbonnier(pred-batch['sharp'].to(device)).mean())
        criterion.set_reference(torch.stack(reference).mean())
    (args.output/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (args.output/'split_manifest.json').write_text(json.dumps({'train':training.keys,'validation':validation.keys},indent=2),encoding='utf-8')
    parameters=sum(p.numel() for p in model.parameters())
    print(f'Train pairs={len(training)}, validation pairs={len(validation)}, parameters={parameters}')
    print('No test images are used for training or checkpoint selection.')
    metrics_path=args.output/'metrics.csv'
    stop=min(cfg['epochs'],args.stop_after_epoch) if args.stop_after_epoch is not None else cfg['epochs']
    if stop<1:parser.error('Stop epoch must be positive.')
    for epoch in range(start,stop):
        model.train();criterion.train()
        sums={};count=0
        for batch in loader:
            blur,target,mask=(batch[k].to(device) for k in ('blur','sharp','mask'))
            optimizer.zero_grad(set_to_none=True)
            pred=model(blur)
            loss,terms=criterion(pred,target,mask,model)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss.')
            loss.backward()
            optimizer.step()
            for key,value in terms.items():sums[key]=sums.get(key,0.0)+value.item()*blur.shape[0]
            count+=blur.shape[0]
        scores=validate(model,val_loader,device)
        improved=scores['psnr']>best
        best=max(best,scores['psnr'])
        lr=optimizer.param_groups[0]['lr']
        scheduler.step()
        payload={'format':FORMAT,'model_config':asdict(model.config),'config':cfg,'model':model.state_dict(),
                 'loss':criterion.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                 'epoch':epoch+1,'best_psnr':best,'validation':scores,'parameters':parameters,
                 'loader_rng':generator.get_state(),'torch_rng':torch.get_rng_state(),
                 'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                 'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        save_checkpoint(args.output/'latest.pt',payload)
        if improved:save_checkpoint(args.output/'best_psnr.pt',payload)
        row={'epoch':epoch+1,'learning_rate':lr,**{k:v/count for k,v in sums.items()},
             'val_psnr':scores['psnr'],'val_ssim':scores['ssim']}
        exists=metrics_path.exists()
        with metrics_path.open('a',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if not exists:writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row))


if __name__=='__main__':main()
