"""Iterative confidence-based compression to a requested logical vertex budget."""
import math
import numpy as np
import torch
from .ising import Compression


def choose_index(logits,selection='confidence'):
    # |logit| is strictly monotone in |sigmoid(logit)-0.5|. Use logits to
    # avoid artificial ties from float32 sigmoid saturation on large graphs.
    if selection=='confidence':
        return int(logits.abs().argmax())
    if selection=='paper_entropy':
        # Literal Algorithm 1 maximizes Bernoulli entropy: least certain edge.
        return int(logits.abs().argmin())
    raise ValueError(selection)


@torch.no_grad()
def compress(problem,reduction,model=None,method='granite',seed=0):
    if not 0<=reduction<1:
        raise ValueError('reduction must be in [0,1)')
    env=Compression(problem); initial=len(env.adj)
    target=max(1,math.ceil(initial*(1-reduction)))
    rng=np.random.default_rng(seed); decisions=[]
    while len(env.adj)>target:
        edges=env.edges()
        if not edges:
            u,v=sorted(env.adj)[:2]; d=1; confidence=None
        elif method in ('granite','paper_entropy','untrained'):
            if model is None:
                raise ValueError('This method requires a model')
            logits=model.predict(env)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Non-finite classifier logits')
            k=choose_index(logits,'paper_entropy' if method=='paper_entropy' else 'confidence')
            u,v,_=edges[k]; d=-1 if logits[k]>=0 else 1
            confidence=float(torch.sigmoid(logits[k]).item())
        elif method=='greedy_weight':
            u,v,w=max(edges,key=lambda e:abs(e[2])); d=1 if w>0 else -1; confidence=None
        elif method=='random':
            u,v,_=edges[int(rng.integers(len(edges)))]; d=int(rng.choice([-1,1])); confidence=None
        else:
            raise ValueError(method)
        env.contract(u,v,d)
        decisions.append({'u':u,'v':v,'direction':d,'probability_anti':confidence})
    return env,decisions
