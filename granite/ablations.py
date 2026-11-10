"""Train loss/depth/width controls on the identical data split with seed zero."""
import argparse
import json
from pathlib import Path
from .train import train


def run(root='results',config_path='configs/desktop.json'):
    root=Path(root);base=json.loads(Path(config_path).read_text());base['seed']=0
    variants={**{f'loss_{name}':{'loss':name} for name in ['bce','mse','softmax']},
              **{f'depth_{depth}':{'layers':depth} for depth in [1,2,4,5]},'width_2':{'width':2}}
    for name,override in variants.items():
        train({**base,**override},root/'data',root/'ablations'/name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',default='results')
    parser.add_argument('--config',default='configs/desktop.json');args=parser.parse_args();run(args.results,args.config)
