"""Joint node/edge GRANITE GNN and confidence-weighted supervised losses."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .ising import Compression


def features(graph):
    nodes=sorted(graph.adj); lookup={v:i for i,v in enumerate(nodes)}; edges=graph.edges()
    x=torch.tensor([[len(graph.adj[v]),sum(graph.adj[v].values()),sum(abs(w) for w in graph.adj[v].values())]
                    for v in nodes],dtype=torch.float32)
    ij=torch.tensor([(lookup[u],lookup[v]) for u,v,w in edges],dtype=torch.long).reshape(-1,2)
    e=torch.tensor([(w,abs(w)) for u,v,w in edges],dtype=torch.float32).reshape(-1,2)
    return x,ij,e


def mlp(inputs,width):
    return nn.Sequential(nn.Linear(inputs,width),nn.ReLU(),nn.Linear(width,width))


class JointLayer(nn.Module):
    def __init__(self,node_dim,edge_dim,width):
        super().__init__()
        self.node=mlp(2*node_dim+edge_dim,width)
        self.edge=mlp(2*node_dim+edge_dim,width)

    def forward(self,h,e,ij):
        u,v=ij.T
        targets=torch.cat([u,v]); neighbors=torch.cat([v,u])
        neighbor_sum=torch.zeros_like(h).index_add(0,targets,h[neighbors])
        edge_sum=h.new_zeros((len(h),e.shape[1])).index_add(0,targets,torch.cat([e,e]))
        new_h=self.node(torch.cat([h,neighbor_sum,edge_sum],dim=1))
        # Symmetrize the paper's ordered concatenation for undirected edges.
        uv=self.edge(torch.cat([h[u],h[v],e],dim=1))
        vu=self.edge(torch.cat([h[v],h[u],e],dim=1))
        return new_h,(uv+vu)/2


class Granite(nn.Module):
    def __init__(self,width=16,layers=3):
        super().__init__()
        if width<1 or layers<1:
            raise ValueError('width and layers must be positive')
        self.width,self.layers=width,layers
        self.blocks=nn.ModuleList([JointLayer(3 if i==0 else width,2 if i==0 else width,width) for i in range(layers)])
        self.readout=nn.Linear(3*width,1,bias=False)

    def forward(self,x,ij,e):
        h=x
        for layer in self.blocks:
            h,e=layer(h,e,ij)
        u,v=ij.T
        # Averaging ordered readouts is invariant to flipping endpoint order.
        return (self.readout(torch.cat([h[u],h[v],e],1))+
                self.readout(torch.cat([h[v],h[u],e],1))).flatten()/2

    def predict(self,graph):
        return self(*features(graph))


def hybrid_loss(logits,labels,lam=0.5,temperature=0.1,kind='hybrid'):
    """Paper's SUM_i [lambda*c_i + (1-lambda)]*BCE_i, per graph.

    y=0 aligned, y=1 anti-aligned, y=-1 neutral (excluded before softmax).
    Confidence weights remain differentiable. They are NOT detached.
    """
    if not 0<=lam<=1 or temperature<=0:
        raise ValueError('Invalid loss hyperparameters')
    valid=labels>=0
    if not valid.any():
        return logits.sum()*0
    z=logits[valid]; y=labels[valid].to(z.dtype); p=z.sigmoid()
    if kind=='mse':
        return ((p-y)**2).sum()
    if kind=='bce':
        lam=0.
    elif kind=='softmax':
        lam=1.
    elif kind!='hybrid':
        raise ValueError(kind)
    confidence=torch.softmax(torch.abs(p-0.5)/temperature,dim=0)
    weights=lam*confidence+(1-lam)
    return (weights*F.binary_cross_entropy_with_logits(z,y,reduction='none')).sum()


def tensor_dataset(records):
    return [(*features(Compression(p)),torch.tensor(labels,dtype=torch.long),record) for p,labels,record in records]


def batch_graphs(records):
    xs=[]; ijs=[]; es=[]; labels=[]; slices=[]; node_offset=0; edge_offset=0
    for x,ij,e,y,_ in records:
        xs.append(x); ijs.append(ij+node_offset); es.append(e); labels.append(y)
        slices.append(slice(edge_offset,edge_offset+len(e)))
        node_offset+=len(x); edge_offset+=len(e)
    return torch.cat(xs),torch.cat(ijs),torch.cat(es),torch.cat(labels),slices


def load_model(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=Granite(saved['width'],saved['layers']); model.load_state_dict(saved['state_dict']); model.eval()
    return model,saved
