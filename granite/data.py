"""Seeded graph generation and exact supervised edge-label datasets."""
import argparse
import json
import time
from pathlib import Path
import networkx as nx
import numpy as np
from .ising import Ising,exact_edge_labels

TOPOLOGIES=('ER','BA','WS')


def generate(n,topology,degree,seed,integer_weights=False):
    if n<2:
        raise ValueError('Require at least two spins')
    if topology=='ER':
        g=nx.erdos_renyi_graph(n,min(degree/(n-1),1.),seed=seed)
    elif topology=='BA':
        g=nx.barabasi_albert_graph(n,max(1,min(int(round(degree/2)),n-1)),seed=seed)
    elif topology=='WS':
        k=min(max(2,int(round(degree))),n-1); k-=k%2
        g=nx.watts_strogatz_graph(n,k,0.3,seed=seed)
    else:
        raise ValueError(topology)
    rng=np.random.default_rng(seed); J=np.zeros((n,n))
    for u,v in sorted(g.edges()):
        J[u,v]=J[v,u]=rng.choice([-2,-1,1,2]) if integer_weights else rng.uniform(-5,5)
    return Ising(J,np.zeros(n))


def make_dataset(output,instances=20,sizes=(4,6,8,10,12),seed=10000):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if (output/'metadata.json').exists():
        raise FileExistsError(f'Dataset already exists: {output}')
    if instances<2:
        raise ValueError('At least two instances per configuration for train/validation split')
    arrays={}; records=[]; index=0; start=time.perf_counter()
    for n in sizes:
        for density,degree in [('sparse',2),('medium',max(2,n//2)),('dense',n-1)]:
            for topology in TOPOLOGIES:
                for instance in range(instances):
                    graph_seed=seed+index
                    problem=generate(n,topology,degree,graph_seed)
                    labels=exact_edge_labels(problem)
                    key=f'g{index:05d}'
                    arrays[key+'_J']=problem.J; arrays[key+'_h']=problem.h
                    arrays[key+'_labels']=labels['labels']; arrays[key+'_edges']=labels['edges']
                    records.append({'id':key,'seed':graph_seed,'n':n,'topology':topology,
                                    'density_setting':density,'requested_degree':degree,
                                    'actual_average_degree':np.count_nonzero(problem.J)/n,
                                    'split':'train' if instance<int(0.8*instances) else 'validation',
                                    'ground_energy':labels['ground_energy'],'ground_states':labels['ground_states'],
                                    'neutral_edges':int(np.sum(labels['labels']==-1))})
                    index+=1
    np.savez_compressed(output/'graphs.npz',**arrays)
    metadata={'records':records,'sizes':list(sizes),'instances_per_configuration':instances,
              'label_tolerance':1e-9,'seed_start':seed,'seconds':time.perf_counter()-start,
              'note':'Exact enumeration of all ground states; no sampled-state label approximation. Topology degree controls are family-specific, not identical exact mean degrees.'}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'graphs':index,'train':sum(r['split']=='train' for r in records),
                      'validation':sum(r['split']=='validation' for r in records),'seconds':metadata['seconds']}),flush=True)


def load_dataset(path,split=None):
    path=Path(path); metadata=json.loads((path/'metadata.json').read_text()); arrays=np.load(path/'graphs.npz',allow_pickle=False)
    result=[]
    for record in metadata['records']:
        if split is None or record['split']==split:
            key=record['id']; result.append((Ising(arrays[key+'_J'],arrays[key+'_h']),arrays[key+'_labels'],record))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/data'); parser.add_argument('--instances',type=int,default=20)
    parser.add_argument('--sizes',type=int,nargs='+',default=[4,6,8,10,12]); parser.add_argument('--seed',type=int,default=10000)
    args=parser.parse_args(); make_dataset(args.output,args.instances,tuple(args.sizes),args.seed)
