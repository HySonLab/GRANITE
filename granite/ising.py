"""Physics-sign Ising Hamiltonians, exhaustive oracles and signed contractions."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Ising:
    """E(s) = -sum_{i<j} J[i,j]s[i]s[j] - h@s + offset.

    Positive coupling favors alignment. J is symmetric; count each edge once.
    This follows GRANITE's minus-sign convention, unlike the RELIC repository.
    """
    J: np.ndarray
    h: np.ndarray
    offset: float = 0.

    def __post_init__(self):
        self.J=np.array(self.J,dtype=np.float64,copy=True)
        self.h=np.array(self.h,dtype=np.float64,copy=True)
        if self.h.ndim!=1 or len(self.h)<1 or self.J.shape!=(len(self.h),len(self.h)):
            raise ValueError('Require square J and a matching nonempty h')
        if not np.isfinite(self.J).all() or not np.isfinite(self.h).all() or not np.isfinite(self.offset):
            raise ValueError('Coefficients must be finite')
        if not np.allclose(self.J,self.J.T) or np.any(np.diag(self.J)!=0):
            raise ValueError('J must be symmetric with zero diagonal')

    @property
    def n(self):
        return len(self.h)

    def energy(self,s):
        s=np.asarray(s)
        if s.shape!=(self.n,) or not np.isin(s,[-1,1]).all():
            raise ValueError('Expected one +/-1 spin per vertex')
        return float(-0.5*s@self.J@s-self.h@s+self.offset)

    def scale(self):
        return max(float(np.abs(self.J).sum()/2+np.abs(self.h).sum()),1.)


def spin_chunks(n,chunk_size=32768):
    for start in range(0,1<<n,chunk_size):
        ids=np.arange(start,min(start+chunk_size,1<<n),dtype=np.uint64)
        yield 2*((ids[:,None]>>np.arange(n,dtype=np.uint64))&1).astype(np.float64)-1


def exact_solve(problem,max_n=22):
    """Exhaustive ground energy and one witness (float64)."""
    if problem.n>max_n:
        raise ValueError(f'Exact enumeration limited to {max_n} spins')
    best,spins=float('inf'),None
    for states in spin_chunks(problem.n):
        energies=-0.5*np.sum((states@problem.J)*states,axis=1)-states@problem.h+problem.offset
        k=int(energies.argmin())
        if energies[k]<best:
            best,spins=float(energies[k]),states[k].astype(np.int8)
    return best,spins


def exact_edge_labels(problem,max_n=18,tolerance=1e-9):
    """Classify against ALL ground states: 0 aligned, 1 anti, -1 neutral.

    Two enumeration passes keep memory bounded, including massively degenerate
    instances. A fixed absolute tolerance defines numerical degeneracy.
    """
    if problem.n>max_n:
        raise ValueError(f'Label generation limited to {max_n} physical spins')
    ground,_=exact_solve(problem,max_n)
    graph=Compression(problem); edges=graph.edges()
    pairs=np.array([(u,v) for u,v,_ in edges],dtype=int).reshape(-1,2)
    seen_aligned=np.zeros(len(edges),bool); seen_anti=np.zeros(len(edges),bool)
    count=0
    for states in spin_chunks(problem.n):
        energies=-0.5*np.sum((states@problem.J)*states,axis=1)-states@problem.h+problem.offset
        ground_states=states[np.abs(energies-ground)<=tolerance]
        count+=len(ground_states)
        if not len(ground_states) or not len(edges):
            continue
        if graph.auxiliary:
            ground_states=np.column_stack([np.ones(len(ground_states)),ground_states])
        products=ground_states[:,pairs[:,0]]*ground_states[:,pairs[:,1]]
        seen_aligned |= (products==1).any(axis=0)
        seen_anti |= (products==-1).any(axis=0)
    labels=np.where(seen_aligned & seen_anti,-1,np.where(seen_anti,1,0)).astype(np.int64)
    return {'edges':pairs,'labels':labels,'ground_energy':ground,'ground_states':count,'tolerance':tolerance}


class Compression:
    """Contract s[v]=d*s[u], preserve restricted energies, record reversible trace.

    The auxiliary field node exists ONLY when h has nonzero entries. Thus the
    zero-field experiments start with exactly n vertices, not n+1.
    """
    def __init__(self,problem):
        self.original=problem
        self.auxiliary=bool(np.any(problem.h))
        shift=int(self.auxiliary)
        self.adj={i:{} for i in range(problem.n+shift)}
        self.offset=float(problem.offset); self.trace=[]
        for i in range(problem.n):
            if problem.h[i]!=0:
                self._edge(0,i+shift,problem.h[i])
            for j in range(i+1,problem.n):
                if problem.J[i,j]!=0:
                    self._edge(i+shift,j+shift,problem.J[i,j])

    def _edge(self,u,v,w):
        if w==0:
            self.adj[u].pop(v,None); self.adj[v].pop(u,None)
        else:
            self.adj[u][v]=float(w); self.adj[v][u]=float(w)

    def edges(self):
        return [(u,v,w) for u in sorted(self.adj) for v,w in sorted(self.adj[u].items()) if u<v]

    def contract(self,u,v,d):
        if u==v or u not in self.adj or v not in self.adj or d not in [-1,1]:
            raise ValueError('Invalid contraction')
        if v not in self.adj[u] and any(self.adj.values()):
            raise ValueError('Nonedge contractions only allowed after all weights vanish')
        self.offset-=d*self.adj[u].get(v,0.)
        for w,weight in list(self.adj[v].items()):
            if w!=u:
                self._edge(u,w,self.adj[u].get(w,0.)+d*weight)
            self.adj[w].pop(v,None)
        del self.adj[v]
        self.trace.append((int(u),int(v),int(d)))

    def reduced(self):
        nodes=sorted(self.adj); index={v:i for i,v in enumerate(nodes)}
        J=np.zeros((len(nodes),len(nodes)))
        for u,v,w in self.edges():
            J[index[u],index[v]]=J[index[v],index[u]]=w
        return Ising(J,np.zeros(len(nodes)),self.offset)

    def lift_augmented(self,s):
        if len(s)!=len(self.adj) or not np.isin(s,[-1,1]).all():
            raise ValueError('Invalid residual spin vector')
        spins=dict(zip(sorted(self.adj),map(int,s)))
        for u,v,d in reversed(self.trace):
            spins[v]=d*spins[u]
        return np.array([spins[i] for i in range(self.original.n+int(self.auxiliary))],dtype=np.int8)

    def lift(self,s):
        augmented=self.lift_augmented(s)
        return augmented[1:]*augmented[0] if self.auxiliary else augmented

    def verify(self,s):
        physical=self.lift(s)
        reduced_energy=self.reduced().energy(s)
        if not np.isclose(self.original.energy(physical),reduced_energy,atol=1e-8,rtol=1e-9):
            raise AssertionError('Contraction/reconstruction energy mismatch')
        return physical,reduced_energy
