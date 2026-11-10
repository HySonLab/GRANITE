"""Exact compression-error study and classical residual-solver experiments."""
import argparse
import hashlib
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from .ising import Compression,exact_solve,exact_edge_labels
from .data import TOPOLOGIES,generate
from .model import load_model
from .compress import compress
from .solvers import simulated_annealing
from .train import seed_all

RATIOS=[0.,.125,.25,.5,.75]


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');writer.writeheader();writer.writerows(rows)


def solution_record(key,name,seed,ratio,env,residual,physical):
    return {'instance':key,'method':name,'seed':seed,'reduction':ratio,'trace':env.trace,
            'residual_spins':residual.tolist(),'spins':physical.tolist(),'offset':env.offset}


@torch.no_grad()
def evaluate(root='results',instances=3,large_instances=2):
    root=Path(root);output=root/'evaluation';output.mkdir(parents=True,exist_ok=True)
    expected=[root/f'seed{s}' for s in range(3)]
    expected += [root/'ablations'/name for name in ['loss_bce','loss_mse','loss_softmax','depth_1','depth_2','depth_4','depth_5','width_2']]
    for run in expected:
        if not (run/'summary.json').exists():
            raise FileNotFoundError(f'Finish training before evaluation: {run}')
    checkpoint_hashes={str(run.relative_to(root))+'/best.pt':hashlib.sha256((run/'best.pt').read_bytes()).hexdigest() for run in expected}
    for seed in range(3):
        name=f'seed{seed}/initial.pt';checkpoint_hashes[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    seed_all(2026);started=time.perf_counter();models=[]
    for seed in range(3):
        trained,_=load_model(root/f'seed{seed}'/'best.pt');initial,_=load_model(root/f'seed{seed}'/'initial.pt')
        models += [('granite',seed,trained,'granite'),('paper_entropy',seed,trained,'paper_entropy'),
                   ('untrained',seed,initial,'untrained'),('random',seed,None,'random')]
    models.append(('greedy_weight',0,None,'greedy_weight'));ablations=[]
    for path in sorted((root/'ablations').glob('*/best.pt')):
        model,_=load_model(path);ablations.append((path.parent.name,0,model,'granite'))
    arrays={};solutions=[];rows=[];classification=[]
    for n in [14,16]:
        for ti,topology in enumerate(TOPOLOGIES):
            for instance in range(instances):
                data_seed=1000000+n*1000+ti*100+instance
                problem=generate(n,topology,4,data_seed);key=f'exact_{topology}_{n}_{instance}'
                arrays[key+'_J']=problem.J;arrays[key+'_h']=problem.h
                oracle=exact_edge_labels(problem);optimum=oracle['ground_energy'];arrays[key+'_labels']=oracle['labels']
                for name,seed,model,method in models+ablations:
                    if model is not None and method!='paper_entropy':
                        z=model.predict(Compression(problem));y=oracle['labels'];valid=y>=0
                        predictions=(z.numpy()>=0).astype(int);top=int(z.abs().argmax()) if len(z) else None
                        classification.append({'instance':key,'n':n,'topology':topology,'method':name,'seed':seed,
                            'correct':int(np.sum(predictions[valid]==y[valid])),'labeled_edges':int(valid.sum()),
                            'neutral_edges':int(np.sum(~valid)),
                            'top_edge_safe':int(y[top]<0 or predictions[top]==y[top]) if top is not None else 1})
                    for ratio in RATIOS:
                        start=time.perf_counter();env,_=compress(problem,ratio,model,method,data_seed+seed)
                        compression_seconds=time.perf_counter()-start;reduced=env.reduced()
                        start=time.perf_counter();energy,residual=exact_solve(reduced);solver_seconds=time.perf_counter()-start
                        physical,verified=env.verify(residual)
                        assert np.isclose(energy,verified) and energy>=optimum-1e-7
                        rows.append({'instance':key,'n':n,'topology':topology,'method':name,'seed':seed,
                            'requested_reduction':ratio,'remaining_vertices':reduced.n,'logical_reduction':1-reduced.n/problem.n,
                            'remaining_edges':len(env.edges()),'energy':energy,'exact_energy':optimum,
                            'absolute_gap':max(0.,energy-optimum),
                            'optimality_percent':100*(1-abs(energy-optimum)/abs(optimum)) if optimum!=0 else '',
                            'preserves_optimum':int(np.isclose(energy,optimum,atol=1e-7,rtol=1e-9)),
                            'compression_seconds':compression_seconds,'solver_seconds':solver_seconds})
                        solutions.append(solution_record(key,name,seed,ratio,env,residual,physical))
        print(f'Exact study n={n} complete in {time.perf_counter()-started:.1f}s',flush=True)
    write_csv(output/'exact.csv',rows);write_csv(output/'classification.csv',classification);large=[]
    large_models=[m for m in models if m[0]=='granite' or (m[0] in ('greedy_weight','random') and m[1]==0)]
    for n in [25,50,100,200,400]:
        for ti,topology in enumerate(TOPOLOGIES):
            for instance in range(large_instances):
                data_seed=2000000+n*1000+ti*100+instance;p=generate(n,topology,4,data_seed)
                key=f'large_{topology}_{n}_{instance}';arrays[key+'_J']=p.J;arrays[key+'_h']=p.h
                reference,reference_spins=simulated_annealing(p,seed=data_seed)
                solutions.append(solution_record(key,'uncompressed_reference',0,0.,Compression(p),reference_spins,reference_spins))
                for name,seed,model,method in large_models:
                    for ratio in RATIOS:
                        start=time.perf_counter();env,_=compress(p,ratio,model,method,data_seed+seed)
                        compression_seconds=time.perf_counter()-start
                        start=time.perf_counter();energy,residual=simulated_annealing(env.reduced(),seed=data_seed)
                        solver_seconds=time.perf_counter()-start;physical,verified=env.verify(residual)
                        assert np.isclose(energy,verified)
                        large.append({'instance':key,'n':n,'topology':topology,'method':name,'seed':seed,
                            'requested_reduction':ratio,'remaining_vertices':len(env.adj),'logical_reduction':1-len(env.adj)/n,
                            'remaining_edges':len(env.edges()),'energy':energy,'reference_sa_energy':reference,
                            'normalized_energy':energy/p.scale(),'normalized_gap_to_sa':(energy-reference)/p.scale(),
                            'compression_seconds':compression_seconds,'solver_seconds':solver_seconds,
                            'total_seconds':compression_seconds+solver_seconds})
                        solutions.append(solution_record(key,name,seed,ratio,env,residual,physical))
        print(f'Classical study n={n} complete in {time.perf_counter()-started:.1f}s',flush=True)
    write_csv(output/'large.csv',large);np.savez_compressed(output/'instances.npz',**arrays)
    (output/'solutions.json').write_text(json.dumps(solutions,separators=(',',':'))+'\n')
    (output/'protocol.json').write_text(json.dumps({'checkpoint_sha256':checkpoint_hashes,'exact_sizes':[14,16],'large_sizes':[25,50,100,200,400],
        'exact_instances_per_configuration':instances,'large_instances_per_configuration':large_instances,
        'logical_reduction_requests':RATIOS,'exact_model_seeds':[0,1,2],'ablation_seeds':[0],
        'annealing':{'reads':16,'sweeps':40,'beta_start':.05,'beta_end':5.,'polish':'best-improving flips on best observed state'},
        'seconds':time.perf_counter()-started,'hardware':'CPU; no QPU or minor embedding; no physical-qubit claims',
        'reference_note':'Large-graph reference is seeded uncompressed simulated annealing, NOT a certified optimum.'},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',default='results')
    parser.add_argument('--instances',type=int,default=3);parser.add_argument('--large-instances',type=int,default=2)
    args=parser.parse_args();evaluate(args.results,args.instances,args.large_instances)
