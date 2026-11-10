"""Generate reproducible tables and plots from measured CSV files."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return list(csv.DictReader(Path(path).open()))


def summary(rows,metric):
    grouped=defaultdict(list)
    for row in rows:
        if row[metric]!='':grouped[row['instance']].append(float(row[metric]))
    values=np.array([np.mean(v) for v in grouped.values()])
    if not len(values):return float('nan'),float('nan')
    return float(values.mean()),float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.


def main(root='results'):
    root=Path(root);out=root/'evaluation';exact=read(out/'exact.csv');large=read(out/'large.csv')
    classification=read(out/'classification.csv');plt.style.use('seaborn-v0_8-whitegrid')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for seed in range(3):
        rows=read(root/f'seed{seed}'/'history.csv');epochs=[int(r['epoch']) for r in rows]
        axes[0].plot(epochs,[float(r['validation_edge_bce']) for r in rows],label=f'Seed {seed}')
        axes[1].plot(epochs,[float(r['validation_accuracy']) for r in rows],label=f'Seed {seed}')
    axes[0].set(xlabel='Epoch',ylabel='Validation BCE per labeled edge',yscale='log')
    axes[1].set(xlabel='Epoch',ylabel='Validation edge accuracy');axes[1].legend()
    fig.suptitle('Supervised GRANITE training on 4–12-spin graphs');fig.tight_layout();fig.savefig(out/'training.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4));ratios=[0.,.125,.25,.5,.75]
    for method in ['granite','paper_entropy','greedy_weight','random','untrained']:
        means=[];errors=[];preservation=[]
        for ratio in ratios:
            rows=[r for r in exact if r['method']==method and float(r['requested_reduction'])==ratio]
            mean,sem=summary(rows,'optimality_percent');means.append(mean);errors.append(sem)
            preservation.append(summary(rows,'preserves_optimum')[0]*100)
        axes[0].errorbar(np.array(ratios)*100,means,yerr=errors,marker='o',label=method)
        axes[1].plot(np.array(ratios)*100,preservation,marker='o',label=method)
    axes[0].set(xlabel='Requested logical reduction (%)',ylabel='Paper optimality metric (%)',title='Exact original and residual solutions, n=14/16')
    axes[1].set(xlabel='Requested logical reduction (%)',ylabel='Runs retaining the original optimum (%)',title='Energy preservation is stricter than high quality')
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'compression.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method in ['granite','greedy_weight','random']:
        means=[];times=[];sizes=[25,50,100,200,400]
        for n in sizes:
            rows=[r for r in large if r['method']==method and int(r['n'])==n and float(r['requested_reduction'])==.75]
            means.append(summary(rows,'normalized_gap_to_sa')[0]);times.append(summary(rows,'total_seconds')[0])
        axes[0].plot(sizes,means,marker='o',label=method);axes[1].plot(sizes,times,marker='o',label=method)
    axes[0].axhline(0,color='gray',linestyle='--');axes[0].set(xlabel='Original spins',ylabel='(Energy − uncompressed SA energy) / scale',title='75% requested logical reduction; lower is better')
    axes[1].set(xlabel='Original spins',ylabel='Compression + residual solve (seconds)',title='CPU pipeline time');axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'large.png',dpi=160);plt.close(fig)
    lines=['# Measured GRANITE reproduction results','',
           'This is a classical CPU reproduction of the compression algorithm. No D-Wave sampling, minor embedding, physical-qubit reduction or quantum advantage was measured.',
           '', '## Training','', '| Seed | Initial validation accuracy | Best-checkpoint accuracy | Best epoch | Epochs | Training seconds |',
           '|---|---:|---:|---:|---:|---:|']
    for seed in range(3):
        s=json.loads((root/f'seed{seed}'/'summary.json').read_text())
        lines.append(f"| {seed} | {s['initial']['accuracy']:.3f} | {s['best']['accuracy']:.3f} | {s['best_epoch']} | {s['epochs_run']} | {s['seconds']:.2f} |")
    lines+=['','![Training](training.png)','','## Held-out edge classification (14/16 spins)','',
            '| Method | Pooled labeled-edge accuracy | First highest-confidence edge safe fraction |','|---|---:|---:|']
    for method in ['granite','untrained','loss_bce','loss_mse','loss_softmax','depth_1','depth_2','depth_4','depth_5','width_2']:
        rows=[r for r in classification if r['method']==method]
        if not rows:continue
        accuracy=sum(int(r['correct']) for r in rows)/sum(int(r['labeled_edges']) for r in rows)
        safe=summary(rows,'top_edge_safe')[0]
        lines.append(f'| {method} | {accuracy:.4f} | {safe:.4f} |')
    lines+=['','## Compression quality with exact residual solves','',
            'Each residual problem is solved exactly, isolating compression loss from solver error. Means average model seeds within each graph before averaging the 18 test graphs. Error bars are one SEM across graphs, not 95% confidence intervals.',
            '', '| Method | Requested reduction | Mean optimality ± SEM (%) | Optimum preserved fraction |','|---|---:|---:|---:|']
    for method in ['granite','paper_entropy','greedy_weight','random','untrained']:
        for ratio in [.125,.25,.5,.75]:
            rows=[r for r in exact if r['method']==method and float(r['requested_reduction'])==ratio]
            mean,sem=summary(rows,'optimality_percent');preserved=summary(rows,'preserves_optimum')[0]
            lines.append(f'| {method} | {100*ratio:g}% | {mean:.2f} ± {sem:.2f} | {preserved:.3f} |')
    lines+=['','![Compression](compression.png)','','## Loss, depth and width ablations','',
            'All controls below use training seed 0 and the same data. The baseline in this table is also seed 0. Checkpoint selection uses each run’s validation objective. These are small-sample controls, not a recreation of the paper’s QPU ablation tables.',
            '', '| Configuration | Validation accuracy | 75% reduction optimality (%) | Optimum preserved fraction |','|---|---:|---:|---:|']
    variants=[('hybrid_depth3_width16',root/'seed0','granite')]+[(p.name,p,p.name) for p in sorted((root/'ablations').iterdir()) if p.is_dir()]
    for name,path,method in variants:
        s=json.loads((path/'summary.json').read_text())
        rows=[r for r in exact if r['method']==method and r['seed']=='0' and float(r['requested_reduction'])==.75]
        lines.append(f"| {name} | {s['best']['accuracy']:.4f} | {summary(rows,'optimality_percent')[0]:.2f} | {summary(rows,'preserves_optimum')[0]:.3f} |")
    lines+=['','## Larger instances: classical solver only','',
            'The reference is an uncompressed simulated-annealing run, NOT the optimum. Negative gaps mean a compressed run beat that heuristic reference. No optimality percentage is computed for these instances.',
            '', '| Original spins | Method | 75% reduction normalized gap to SA | Mean pipeline seconds |','|---|---|---:|---:|']
    for n in [25,50,100,200,400]:
        for method in ['granite','greedy_weight','random']:
            rows=[r for r in large if r['method']==method and int(r['n'])==n and float(r['requested_reduction'])==.75]
            lines.append(f"| {n} | {method} | {summary(rows,'normalized_gap_to_sa')[0]:.4f} | {summary(rows,'total_seconds')[0]:.4f} |")
    lines+=['','![Larger cases](large.png)','','## Interpretation','',
            '- `granite` selects the most confident edge, following the prose. `paper_entropy` follows the maximum-entropy selection printed in Algorithm 1. This is an explicit ambiguity comparison, not a silent correction.',
            '- High individual-edge classification accuracy does not guarantee safe repeated contraction. Earlier errors restrict subsequent possibilities, and residual graphs differ from the training distribution.',
            '- `greedy_weight` satisfies the largest-magnitude residual edge and is an important inexpensive control. Its strength limits any claim that learning is necessary on these synthetic graphs.',
            '- Reduction is the fraction of logical augmented vertices removed, rounded to an integer target. Edge counts are recorded separately. Neither quantity is a measurement of physical qubits.',
            '- The large-graph solver uses 16 Metropolis chains and 40 sweeps, then local polish. Fixed reads/sweeps are not fixed wall time; compression and solver times are recorded separately.',
            '- Original paper training used much larger labeled data, and its experiments used D-Wave and Gurobi. Here training uses 720 graphs plus 180 validation graphs, all 4–12 spins; exact tests hold out both seeds and sizes.',
            '', 'Artifacts: [exact CSV](exact.csv), [large CSV](large.csv), [classification CSV](classification.csv), [instances](instances.npz), [traces and solutions](solutions.json), [protocol](protocol.json), [verification](verification.json).','']
    (out/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',default='results')
    main(parser.parse_args().results)
