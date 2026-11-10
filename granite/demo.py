"""Compress a tiny biased problem and export a solver-neutral residual model."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .ising import Ising,exact_solve
from .model import load_model
from .compress import compress


def export_residual(env):
    """D-Wave/dimod convention is +sum J*s*s +sum h*s +offset: negate ours."""
    reduced=env.reduced()
    return {'energy_convention':'sum_{i<j} J[i,j]*s[i]*s[j] + sum_i h[i]*s[i] + offset',
            'linear':(-reduced.h).tolist(),
            'quadratic':[[int(i),int(j),float(-reduced.J[i,j])] for i in range(reduced.n)
                         for j in range(i+1,reduced.n) if reduced.J[i,j]!=0],
            'offset':reduced.offset,'residual_vertex_order':sorted(env.adj),'contraction_trace':env.trace,
            'auxiliary_node_used':env.auxiliary,'original_physical_spins':env.original.n}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='results/seed0/best.pt');parser.add_argument('--export')
    args=parser.parse_args();torch.set_num_threads(1)
    problem=Ising([[0,2,-1],[2,0,1],[-1,1,0]],[.4,-.6,.2],1.5)
    model,_=load_model(args.checkpoint);env,decisions=compress(problem,.5,model)
    energy,s=exact_solve(env.reduced());physical,_=env.verify(s)
    print('Original physical spins:',problem.n,'; augmented graph vertices:',problem.n+int(env.auxiliary))
    print('Contractions:',env.trace);print('Remaining augmented vertices:',len(env.adj))
    print('Reconstructed spins:',physical.tolist(),'energy:',energy,'original optimum:',exact_solve(problem)[0])
    if args.export:
        path=Path(args.export);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(export_residual(env),indent=2)+'\n');print('Exported:',path)


if __name__=='__main__':main()
