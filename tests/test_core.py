import itertools
import numpy as np
import pytest
import torch
from granite.ising import Ising,Compression,exact_solve,exact_edge_labels
from granite.data import generate
from granite.model import Granite,features,hybrid_loss,batch_graphs,tensor_dataset
from granite.compress import compress,choose_index


def test_physics_sign_convention():
    p=Ising([[0,2],[2,0]],[0,0])
    assert p.energy([1,1])==-2
    assert p.energy([1,-1])==2
    assert exact_edge_labels(p)['labels'].tolist()==[0]
    p.J*=-1
    assert exact_edge_labels(p)['labels'].tolist()==[1]


def test_all_ground_states_not_just_one_witness():
    p=Ising(-(np.ones((3,3))-np.eye(3)),np.zeros(3))
    labels=exact_edge_labels(p)
    assert labels['ground_states']==6
    assert labels['ground_energy']==-1
    assert labels['labels'].tolist()==[-1,-1,-1]
    # Either direction on one neutral edge retains at least one ground state.
    for d in [-1,1]:
        graph=Compression(p); graph.contract(0,1,d)
        assert exact_solve(graph.reduced())[0]==-1


def test_all_degenerate_states_counted_in_empty_graph():
    p=Ising(np.zeros((5,5)),np.zeros(5))
    labels=exact_edge_labels(p)
    assert labels['ground_states']==32 and labels['labels'].size==0


@pytest.mark.parametrize('d',[-1,1])
def test_partial_contraction_preserves_every_restricted_energy(d):
    p=generate(6,'ER',3,13);p.h=np.array([.4,-1.,0.,.2,0.,0.]);p.offset=2.7
    graph=Compression(p)
    graph.contract(1,0,d) # Absorb auxiliary node to test gauge correction.
    u,v,_=graph.edges()[0];graph.contract(u,v,-1)
    for s in itertools.product([-1,1],repeat=len(graph.adj)):
        lifted,e=graph.verify(s)
        assert e==pytest.approx(p.energy(lifted))


@pytest.mark.parametrize('seed',range(6))
def test_random_compression_reconstruction(seed):
    p=generate(8,'ER',2,seed)
    graph,_=compress(p,.75,method='random',seed=seed)
    e,s=exact_solve(graph.reduced());lifted,energy=graph.verify(s)
    assert energy==pytest.approx(e)
    assert energy>=exact_solve(p)[0]-1e-8
    assert len(graph.adj)==2


def test_cancellation_and_isolated_nodes():
    p=Ising([[0,1,2,0],[1,0,-2,0],[2,-2,0,0],[0,0,0,0]],[0]*4)
    graph=Compression(p);graph.contract(0,1,1)
    assert graph.edges()==[]
    graph.contract(0,2,1);graph.contract(0,3,1)
    assert graph.verify([1])[1]==-1
    graph,_=compress(Ising(np.zeros((8,8)),np.zeros(8)),.75,method='greedy_weight')
    assert len(graph.adj)==2


def test_wrong_alignment_can_lose_optimum():
    p=Ising([[0,1],[1,0]],[0,0]);graph=Compression(p);graph.contract(0,1,-1)
    assert exact_solve(p)[0]==-1 and exact_solve(graph.reduced())[0]==1


def test_neutral_mask_hybrid_endpoints_and_differentiable_weights():
    z=torch.tensor([-1.2,.3,2.],requires_grad=True);y=torch.tensor([0,1,-1])
    bce=torch.nn.functional.binary_cross_entropy_with_logits(z[:2],y[:2].float(),reduction='none')
    c=torch.softmax(abs(z[:2].sigmoid()-.5)/.1,0)
    torch.testing.assert_close(hybrid_loss(z,y,lam=0),bce.sum())
    torch.testing.assert_close(hybrid_loss(z,y,lam=1),(c*bce).sum())
    loss=hybrid_loss(z,y);loss.backward()
    assert z.grad[2]==0 and torch.isfinite(z.grad).all()
    torch.testing.assert_close(hybrid_loss(z,torch.full_like(y,-1)),z.sum()*0)


def test_hybrid_loss_gradient_finite_difference():
    z=torch.tensor([-.7,.2,1.3],dtype=torch.float64,requires_grad=True);y=torch.tensor([0,1,1])
    gradient=torch.autograd.grad(hybrid_loss(z,y),z)[0]
    numeric=[]
    for i in range(3):
        plus=z.detach().clone();minus=z.detach().clone();plus[i]+=1e-5;minus[i]-=1e-5
        numeric.append((hybrid_loss(plus,y)-hybrid_loss(minus,y))/2e-5)
    torch.testing.assert_close(gradient,torch.stack(numeric),atol=1e-7,rtol=1e-5)


def test_entropy_algorithm_and_confidence_prose_are_opposites():
    z=torch.tensor([.1,-4.,2.])
    assert choose_index(z,'confidence')==1
    assert choose_index(z,'paper_entropy')==0


def test_equivariance_and_disjoint_batching():
    torch.manual_seed(4);model=Granite(8,3)
    p=generate(8,'BA',4,2);x,ij,e=features(Compression(p))
    perm=torch.randperm(len(x));inverse=torch.argsort(perm)
    torch.testing.assert_close(model(x,ij,e),model(x[perm],inverse[ij.flip(1)],e),atol=1e-5,rtol=1e-5)
    labels=exact_edge_labels(p)['labels'];records=tensor_dataset([(p,labels,{}),(p,labels,{})])
    bx,bij,be,_,slices=batch_graphs(records);result=model(bx,bij,be)
    torch.testing.assert_close(result[slices[0]],model(x,ij,e))
    torch.testing.assert_close(result[slices[1]],model(x,ij,e))


def test_gradient_updates_node_and_edge_mlps():
    torch.manual_seed(7);model=Granite(8,3);p=generate(8,'ER',3,4)
    labels=torch.tensor(exact_edge_labels(p)['labels'])
    loss=hybrid_loss(model.predict(Compression(p)),labels);loss.backward()
    assert model.blocks[0].node[0].weight.grad.abs().sum()>0
    assert model.blocks[0].edge[0].weight.grad.abs().sum()>0


def test_target_rounding_and_no_compression():
    p=generate(9,'ER',3,3)
    graph,_=compress(p,.125,method='greedy_weight')
    assert len(graph.adj)==8
    graph,_=compress(p,0,method='greedy_weight')
    assert graph.trace==[]
    np.testing.assert_array_equal(graph.reduced().J,p.J)


def test_annealing_energy_and_seed_reproducibility():
    from granite.solvers import simulated_annealing
    p=generate(8,'ER',4,31);p.h=np.array([.3,-.2,.5,0,0,0,0,0])
    energy,spins=simulated_annealing(p,seed=5)
    repeated,other=simulated_annealing(p,seed=5)
    assert energy==pytest.approx(p.energy(spins))
    assert energy>=exact_solve(p)[0]-1e-8
    assert energy==repeated
    np.testing.assert_array_equal(spins,other)
    # A local optimum after the documented polish.
    assert np.min(2*spins*(p.J@spins+p.h))>=-1e-8


def test_checkpoint_roundtrip(tmp_path):
    from granite.model import load_model
    model=Granite(8,3);p=generate(6,'BA',2,9)
    path=tmp_path/'model.pt'
    torch.save({'state_dict':model.state_dict(),'width':8,'layers':3},path)
    loaded,_=load_model(path)
    torch.testing.assert_close(model.predict(Compression(p)),loaded.predict(Compression(p)))


def test_supervised_training_smoke(tmp_path):
    from granite.data import make_dataset
    from granite.train import train
    make_dataset(tmp_path/'data',instances=2,sizes=(4,))
    config={'seed':0,'width':4,'layers':2,'loss':'hybrid','lambda':.5,'temperature':.1,
            'lr':.001,'batch_size':4,'epochs':2,'patience':2,'gradient_clip':10.}
    result=train(config,tmp_path/'data',tmp_path/'run')
    assert np.isfinite(result['best']['loss'])
    assert (tmp_path/'run'/'best.pt').exists()
    with pytest.raises(FileExistsError):train(config,tmp_path/'data',tmp_path/'run')


def test_export_positive_sign_solver_convention():
    from granite.demo import export_residual
    p=generate(5,'ER',3,7);p.h=np.array([.5,-.2,0,0,0])
    env,_=compress(p,.5,method='greedy_weight');data=export_residual(env)
    for s in itertools.product([-1,1],repeat=len(env.adj)):
        exported=sum(w*s[u]*s[v] for u,v,w in data['quadratic'])+np.dot(data['linear'],s)+data['offset']
        assert exported==pytest.approx(p.energy(env.lift(s)))
