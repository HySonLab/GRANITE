"""Supervised learning from exact all-ground-state edge labels (not RL)."""
import argparse
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import load_dataset
from .model import Granite,batch_graphs,tensor_dataset,hybrid_loss


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)


def objective(logits,labels,slices,config):
    return torch.stack([hybrid_loss(logits[s],labels[s],config['lambda'],config['temperature'],config['loss'])
                        for s in slices]).mean()


@torch.no_grad()
def validate(model,records,config):
    model.eval(); total_loss=0.; bce=0.; correct=0; count=0
    for start in range(0,len(records),config['batch_size']):
        selected=records[start:start+config['batch_size']]
        x,ij,e,labels,slices=batch_graphs(selected); z=model(x,ij,e)
        total_loss+=float(objective(z,labels,slices,config))*len(selected)
        valid=labels>=0
        if valid.any():
            bce+=float(F.binary_cross_entropy_with_logits(z[valid],labels[valid].float(),reduction='sum'))
            correct+=int(((z[valid]>=0)==labels[valid]).sum()); count+=int(valid.sum())
    return {'loss':total_loss/len(records),'edge_bce':bce/max(count,1),'accuracy':correct/max(count,1),'valid_edges':count}


def train(config,data,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if (output/'history.csv').exists():
        raise FileExistsError(f'Will not overwrite an existing run: {output}')
    seed_all(config['seed'])
    training=tensor_dataset(load_dataset(data,'train')); validation=tensor_dataset(load_dataset(data,'validation'))
    model=Granite(config['width'],config['layers'])
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    start=time.perf_counter(); initial=validate(model,validation,config)
    best=initial['loss']; best_epoch=0; stale=0; rows=[]
    def save(name,epoch,score):
        torch.save({'state_dict':model.state_dict(),'width':model.width,'layers':model.layers,
                    'epoch':epoch,'validation_loss':score,'config':config},output/name)
    save('initial.pt',0,best); save('best.pt',0,best)
    rows.append({'epoch':0,'train_loss':'','validation_loss':best,'validation_edge_bce':initial['edge_bce'],
                 'validation_accuracy':initial['accuracy'],'elapsed_seconds':time.perf_counter()-start})
    rng=np.random.default_rng(config['seed'])
    for epoch in range(1,config['epochs']+1):
        model.train(); total=0.; order=rng.permutation(len(training))
        for start_idx in range(0,len(order),config['batch_size']):
            records=[training[i] for i in order[start_idx:start_idx+config['batch_size']]]
            x,ij,e,labels,slices=batch_graphs(records)
            optimizer.zero_grad(); logits=model(x,ij,e); loss=objective(logits,labels,slices,config)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
            optimizer.step(); total+=float(loss.detach())*len(records)
        metrics=validate(model,validation,config)
        if metrics['loss']<best-1e-8:
            best=metrics['loss'];best_epoch=epoch;stale=0;save('best.pt',epoch,best)
        else:
            stale+=1
        rows.append({'epoch':epoch,'train_loss':total/len(training),'validation_loss':metrics['loss'],
                     'validation_edge_bce':metrics['edge_bce'],'validation_accuracy':metrics['accuracy'],
                     'elapsed_seconds':time.perf_counter()-start})
        with (output/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');writer.writeheader();writer.writerows(rows)
        if epoch%10==0 or epoch==1:
            print(json.dumps({'run':output.name,**rows[-1]}),flush=True)
        if stale>=config['patience']:
            break
    save('last.pt',epoch,metrics['loss'])
    saved=torch.load(output/'best.pt',weights_only=True);model.load_state_dict(saved['state_dict'])
    final=validate(model,validation,config)
    summary={'initial':initial,'best':final,'best_epoch':best_epoch,'epochs_run':epoch,
             'training_graphs':len(training),'validation_graphs':len(validation),'seconds':time.perf_counter()-start}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/desktop.json');parser.add_argument('--data',default='results/data')
    parser.add_argument('--output',required=True);parser.add_argument('--seed',type=int)
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    if args.seed is not None:config['seed']=args.seed
    train(config,args.data,args.output)
