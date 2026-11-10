"""Recompute exact labels, replay every trace, and verify saved result metrics."""
import argparse
import hashlib
import torch
import csv
import json
from pathlib import Path
import numpy as np
from .ising import Ising,Compression,exact_solve,exact_edge_labels
from .data import load_dataset
from .model import load_model


def verify(root='results'):
    root=Path(root);label_graphs=0;torch.set_num_threads(1)
    protocol=json.loads((root/'evaluation'/'protocol.json').read_text())
    for path,digest in protocol['checkpoint_sha256'].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,path
    for p,labels,record in load_dataset(root/'data'):
        actual=exact_edge_labels(p)
        np.testing.assert_array_equal(labels,actual['labels'])
        assert np.isclose(record['ground_energy'],actual['ground_energy'])
        assert record['ground_states']==actual['ground_states'];label_graphs+=1
    out=root/'evaluation';arrays=np.load(out/'instances.npz',allow_pickle=False)
    solutions=json.loads((out/'solutions.json').read_text())
    lookup={};oracles={};reduced_oracles={};count=0
    for solution in solutions:
        key=solution['instance'];p=Ising(arrays[key+'_J'],arrays[key+'_h']);env=Compression(p)
        for u,v,d in solution['trace']:env.contract(u,v,d)
        physical,energy=env.verify(solution['residual_spins'])
        np.testing.assert_array_equal(physical,solution['spins'])
        assert np.isclose(env.offset,solution['offset'])
        identity=(key,solution['method'],str(solution['seed']),float(solution['reduction']))
        assert identity not in lookup
        lookup[identity]=(energy,len(env.adj),len(env.edges()))
        if key.startswith('exact_'):
            if key not in oracles:
                oracle=exact_edge_labels(p);oracles[key]=oracle['ground_energy']
                np.testing.assert_array_equal(arrays[key+'_labels'],oracle['labels'])
            trace_key=(key,tuple(tuple(t) for t in solution['trace']))
            if trace_key not in reduced_oracles:
                reduced_oracles[trace_key]=exact_solve(env.reduced())[0]
            assert np.isclose(energy,reduced_oracles[trace_key])
        count+=1
    csv_count=0
    for filename in ['exact.csv','large.csv']:
        for row in csv.DictReader((out/filename).open()):
            key=row['instance'];identity=(key,row['method'],row['seed'],float(row['requested_reduction']))
            energy,vertices,edges=lookup[identity]
            assert np.isclose(energy,float(row['energy']))
            assert vertices==int(row['remaining_vertices']) and edges==int(row['remaining_edges'])
            assert np.isclose(1-vertices/int(row['n']),float(row['logical_reduction']))
            if filename=='exact.csv':
                optimum=oracles[key]
                assert np.isclose(optimum,float(row['exact_energy']))
                assert np.isclose(max(0.,energy-optimum),float(row['absolute_gap']))
                assert int(np.isclose(energy,optimum,atol=1e-7,rtol=1e-9))==int(row['preserves_optimum'])
                if row['optimality_percent']:
                    assert np.isclose(100*(1-abs(energy-optimum)/abs(optimum)),float(row['optimality_percent']))
            else:
                p=Ising(arrays[key+'_J'],arrays[key+'_h'])
                baseline=lookup[(key,'uncompressed_reference','0',0.)][0]
                assert np.isclose(baseline,float(row['reference_sa_energy']))
                assert np.isclose((energy-baseline)/p.scale(),float(row['normalized_gap_to_sa']))
            csv_count+=1
    classification_rows=0;models={}
    for row in csv.DictReader((out/'classification.csv').open()):
        method,seed=row['method'],row['seed'];identity=(method,seed)
        if identity not in models:
            path=root/f'seed{seed}'/('initial.pt' if method=='untrained' else 'best.pt') if method in ['granite','untrained'] else root/'ablations'/method/'best.pt'
            models[identity]=load_model(path)[0]
        p=Ising(arrays[row['instance']+'_J'],arrays[row['instance']+'_h'])
        with torch.no_grad():z=models[identity].predict(Compression(p))
        y=arrays[row['instance']+'_labels'];valid=y>=0;predicted=(z.numpy()>=0).astype(int)
        assert int(np.sum(predicted[valid]==y[valid]))==int(row['correct'])
        assert int(valid.sum())==int(row['labeled_edges'])
        assert int((~valid).sum())==int(row['neutral_edges'])
        top=int(z.abs().argmax()) if len(z) else None
        safe=int(y[top]<0 or predicted[top]==y[top]) if top is not None else 1
        assert safe==int(row['top_edge_safe']);classification_rows+=1
    summary={'classification_rows_verified':classification_rows,'checkpoint_hashes_verified':len(protocol['checkpoint_sha256']),'status':'passed','exact_training_and_validation_labels_recomputed':label_graphs,
             'solution_traces_verified':count,'csv_rows_verified':csv_count,
             'original_test_exact_oracles_recomputed':len(oracles),
             'distinct_reduced_exact_oracles_recomputed':len(reduced_oracles)}
    (out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',default='results')
    verify(parser.parse_args().results)
