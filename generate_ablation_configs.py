"""Write separate configurations for structural and loss ablations."""
import argparse
import copy
import itertools
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path('configs/wtb_cdd.json'))
    parser.add_argument('--output',type=Path,default=Path('configs/ablations'))
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Output directory is not empty.')
    final=json.loads(args.base.read_text(encoding='utf-8'))
    structural=copy.deepcopy(final)
    structural['loss']['weighted_pixel']=False
    structural['loss']['kernel_weight']=0.0
    configs={}
    for name,spatial,frequency,guided in [('base',False,False,False),('dsmee',True,False,False),
                                          ('structure_full',True,True,True),('frequency_unguided',True,True,False)]:
        cfg=copy.deepcopy(structural)
        cfg['model'].update(use_dsmee=spatial,use_dfem=frequency,edge_guidance=guided)
        configs[name]=cfg
    branches=('small','medium','large')
    for size in (1,2,3):
        for selected in itertools.combinations(branches,size):
            cfg=copy.deepcopy(structural)
            cfg['model']['branches']=list(selected)
            configs['branches_'+'_'.join(selected)]=cfg
    for weighted,kernel in itertools.product((False,True),repeat=2):
        cfg=copy.deepcopy(final)
        cfg['loss']['weighted_pixel']=weighted
        cfg['loss']['kernel_weight']=final['loss']['kernel_weight'] if kernel else 0.0
        configs[f'loss_weighted_{int(weighted)}_kernel_{int(kernel)}']=cfg
    args.output.mkdir(parents=True,exist_ok=True)
    for name,cfg in configs.items():(args.output/(name+'.json')).write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    print(f'Wrote {len(configs)} configurations. Train each independently from scratch.')


if __name__=='__main__':main()
