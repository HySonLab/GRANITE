"""Local CPU solver for residual problems; no quantum-service calls."""
import numpy as np


def simulated_annealing(problem,seed=0,reads=16,sweeps=40,beta_start=.05,beta_end=5.):
    """Metropolis chains; return best observed state with local-search polish.

    Fixed reads/sweeps before and after compression are not equal wall time.
    """
    if reads<1 or sweeps<1 or beta_start<=0 or beta_end<beta_start:
        raise ValueError('Invalid annealing schedule')
    rng=np.random.default_rng(seed);spins=rng.choice([-1,1],size=(reads,problem.n))
    field=spins@problem.J+problem.h
    energies=-.5*np.sum((spins@problem.J)*spins,axis=1)-spins@problem.h+problem.offset
    k=int(energies.argmin());best_e=float(energies[k]);best_s=spins[k].copy()
    for beta in np.geomspace(beta_start,beta_end,sweeps):
        for vertex in rng.permutation(problem.n):
            delta=2*spins[:,vertex]*field[:,vertex]
            indices=np.flatnonzero(np.log(rng.random(reads)) < -beta*delta)
            if len(indices):
                old=spins[indices,vertex].copy();spins[indices,vertex]*=-1
                field[indices]-=2*old[:,None]*problem.J[vertex];energies[indices]+=delta[indices]
                k=int(energies.argmin())
                if energies[k]<best_e:
                    best_e=float(energies[k]);best_s=spins[k].copy()
    while True:
        delta=2*best_s*(problem.J@best_s+problem.h);k=int(delta.argmin())
        if delta[k]>=-1e-10:break
        best_s[k]*=-1
    return problem.energy(best_s),best_s.astype(np.int8)
