# GRANITE: Scalable Quantum-Inspired Optimization Through Dynamic Qubit Compression

A student-friendly, independent reproduction of **GRANITE**, the method in Co Tran, Quoc-Bao Tran, Hy Truong Son, and Thang N. Dinh, *AAAI 2025*, pp. 11353–11361. [Published paper](https://doi.org/10.1609/aaai.v39i11.33235) · [arXiv version](https://arxiv.org/abs/2412.18571)

**What is reproduced:** exact ground-state edge labels, the joint node/edge GNN, the hybrid supervised loss, iterative merge/flip-merge compression, residual solving, reconstruction, and small controlled experiments. This is newly written code based on the supplied nine-page paper, not recovered author code.

**What is not reproduced:** the original 97,500-graph dataset, D-Wave sampling, Pegasus embeddings, physical-qubit savings, Gurobi runs, or the numerical results in the original tables. The supplied study runs on a CPU and makes those limits explicit.

Start with the [measured results](results/evaluation/REPORT.md). Code, exactly labeled training/validation graphs, eleven trained models, initial/best/final checkpoints, raw test instances, contraction traces, reconstructed spins, CSVs, and plots are included.

## 1. The idea, and how it differs from RELIC

An Ising problem assigns a spin −1 or +1 to each vertex. GRANITE learns whether two neighboring spins agree or disagree **at the global minimum of the energy**. If the prediction is correct, the two variables can be combined into one without losing the optimal energy. Repeating this process produces a smaller problem that can be given to another solver.

The workflow is:

```text
Small training graphs → exact ground states → edge labels → supervised GNN

New graph → score edges → contract one edge → score the changed graph again
          → stop at target size → solve residual problem → reconstruct spins
```

GRANITE is a **supervised classifier**, not a reinforcement-learning policy. RELIC learns compression decisions from terminal rewards; GRANITE learns edge alignment from exact labels and stops at a configurable partial reduction. It is therefore useful to study them together, but their training objectives and inference pipelines should not be confused.

This package is standalone: it does not import or require the RELIC repository.

## 2. Quick start

Python 3.10 or later is required. No GPU, PyTorch Geometric, D-Wave credentials, or commercial solver is needed.

```bash
git clone https://github.com/HySonLab/GRANITE.git
cd GRANITE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m pytest -q
python -m granite.demo
```

If suitable PyTorch is already installed, the explicit CPU-wheel installation can be omitted. The [tested package versions](requirements-tested.txt) and [execution environment](results/environment.json) record the actual run. Platform-independent minimum dependencies are in `pyproject.toml`.

### Verify the included results

```bash
python -m granite.verify
python -m granite.report
```

Verification recomputes all 900 training/validation label sets, replays every saved contraction trace, checks reconstructed energies and CSV metrics, and independently solves the small original and residual problems exactly. Plot generation uses the raw measured CSVs.

### Repeat the complete desktop study

```bash
bash scripts/reproduce.sh runs/my-study
```

The script generates a fresh dataset, trains three main seeds and eight ablations, evaluates, verifies, and generates the report. It writes to a new destination, leaving the committed `results/` intact. Existing datasets and training histories are not overwritten; use a new directory for a new experiment. The published main training runs ran concurrently in separate processes, whereas this script runs them sequentially. Timing will vary with hardware and desktop activity.

### Run the pieces separately

```bash
python -m granite.data --output runs/example/data
python -m granite.train --data runs/example/data --seed 0 --output runs/example/seed0
python -m granite.train --data runs/example/data --seed 1 --output runs/example/seed1
python -m granite.train --data runs/example/data --seed 2 --output runs/example/seed2
python -m granite.ablations --results runs/example
python -m granite.evaluate --results runs/example
python -m granite.verify --results runs/example
python -m granite.report --results runs/example
```

Evaluation requires all eleven completed training runs and records the exact checkpoint hashes used. The examples use a common configuration, `configs/desktop.json`.

For a shorter exercise, generate a smaller dataset and train one model:

```bash
python -m granite.data --output runs/tiny/data --sizes 4 6 --instances 5
python -m granite.train --data runs/tiny/data --output runs/tiny/model
python -m granite.demo --checkpoint runs/tiny/model/best.pt
```

### Use your own Ising problem

```python
import numpy as np
from granite.ising import Ising, exact_solve
from granite.model import load_model
from granite.compress import compress

# GRANITE convention: E = -sum_{i<j} J[i,j]*s[i]*s[j] - h@s + offset
problem = Ising(
    J=np.array([[0.0, 2.0, -1.0], [2.0, 0.0, 1.0], [-1.0, 1.0, 0.0]]),
    h=np.array([0.4, -0.6, 0.2]),
)
model, checkpoint = load_model("results/seed0/best.pt")
compressed, decisions = compress(problem, reduction=0.5, model=model)
reduced_problem = compressed.reduced()
energy, residual_spins = exact_solve(reduced_problem)
physical_spins, checked_energy = compressed.verify(residual_spins)
print(physical_spins, energy)
```

`exact_solve` deliberately refuses problems larger than its default limit of 22 spins. Use `granite.solvers.simulated_annealing` for larger residuals; its returned energy is heuristic, not an optimality certificate. Checkpoints are inference snapshots, not optimizer/RNG snapshots for exact training resume.

## 3. Repository guide

| File | Purpose |
|---|---|
| `granite/ising.py` | Energy convention, exhaustive oracles, all-ground-state labels, signed contractions and lifting |
| `granite/data.py` | Seeded ER/BA/WS generation and dataset serialization |
| `granite/model.py` | Node/edge features, joint GNN, hybrid loss and graph batching |
| `granite/train.py` | Supervised training, validation and checkpoint selection |
| `granite/compress.py` | Iterative inference, target size, confidence/entropy modes and baselines |
| `granite/solvers.py` | Classical simulated annealing with local-search polish |
| `granite/ablations.py` | Loss, depth and hidden-width controls |
| `granite/evaluate.py` | Exact small-graph and larger heuristic experiments |
| `granite/verify.py` | Independent recomputation of labels, traces and metrics |
| `granite/report.py` | Tables and plots from measured data |
| `granite/demo.py` | Small example and residual-model export |
| `tests/test_core.py` | Algebra, neutrality, symmetry, gradients and pipeline tests |
| `results/data/` | Actual 720 training and 180 validation graphs and exact labels |
| `results/seed0/`, `seed1/`, `seed2/` | Main models and training histories |
| `results/ablations/` | Eight additional trained configurations |
| `results/evaluation/` | Test instances, predictions, contraction traces and measured reports |

## 4. Energy convention and contraction algebra

We follow the paper's **negative-sign** Hamiltonian, counting each undirected edge once:

```text
E(s) = -sum_{i<j} J[i,j]*s[i]*s[j] - sum_i h[i]*s[i] + offset.
```

`J` is symmetric with zero diagonal. The implementation uses `-0.5*s @ J @ s` to avoid double counting. A positive coupling favors equal spins; a negative coupling favors opposite spins. **This convention is opposite to the positive-sign convention in the RELIC reproduction.** Converting between them requires negating `J` and `h`, not the offset.

A contraction imposes `s[v] = d*s[u]`, with `d=+1` for merge and `d=-1` for flip-merge. Substituting gives:

```text
offset_new = offset_old - d*J[u,v]
J_new[u,w] = J_old[u,w] + d*J_old[v,w]
```

Then delete `v`. Missing edges count as zero; new redirected edges are created; exact cancellations are removed. This is an energy identity **restricted to assignments satisfying the imposed relation**. If the relation excludes every original ground state, the optimum is lost. The code tracks the offset explicitly even though the paper's contraction pseudocode does not show it.

### Local fields and reconstruction

For nonzero fields, add auxiliary vertex 0 with edge weights `h[i]` to physical vertex `i+1`. Original physical spins satisfy `s[i] = z[i+1]*z[0]`. This represents fields using pair interactions without needing to pin the auxiliary spin during solving.

The auxiliary vertex may be merged. After solving the residual graph, reverse every recorded relationship to reconstruct all augmented spins; then multiply by the reconstructed auxiliary spin to recover physical spins. `compressed.verify(...)` checks the original and residual energies agree.

For zero-field problems, the unused auxiliary vertex is omitted, so an experimental `n`-spin graph starts with exactly `n` vertices. With nonzero fields, the requested budget counts augmented vertices, including the auxiliary variable. These counts are logical variables, not physical hardware qubits.

### Disconnected and edgeless graphs

The model scores existing nonzero edges. If every residual edge vanishes before reaching the target, the remaining variables do not interact. The code joins them deterministically with zero-energy merges. This completes the requested size reduction without changing energy. It also handles initially isolated vertices and cancellation during contraction.

## 5. Exact edge labels: why one optimal solution is insufficient

For each edge, inspect **all** ground states:

- Label `0`: the endpoints are aligned in every ground state.
- Label `1`: the endpoints are anti-aligned in every ground state.
- Label `-1`: neutral; both relationships occur among ground states.

A single optimal witness cannot distinguish persistent alignment from a neutral edge. For example, an antiferromagnetic triangle has six ground states, and all three edges are neutral. A test checks this example explicitly.

The labeling routine enumerates spin assignments in chunks twice: first to find the minimum energy, then to collect the relationships seen among all states within an absolute energy tolerance of `1e-9`. It does not retain every ground-state vector, so highly degenerate cases remain memory bounded. The default labeling limit is 18 physical spins; the desktop data only need up to 12.

An individual neutral edge permits either contraction while retaining at least one optimum. This does **not** mean several relationships selected independently from the original graph are jointly compatible. GRANITE predicts again after each contraction. Our exact tests check complete resulting subproblems rather than assuming individual decisions compose safely.

## 6. The joint node/edge GNN

The feature list follows the method section:

```text
node: [degree, signed weighted degree, absolute weighted degree]
edge: [weight, absolute weight]
```

No degree normalization, learned node identifiers, attention layer, or reinforcement-learning reward is introduced. “Transformer” is the method name; this architecture is a message-passing GNN, not a standard self-attention transformer.

For each layer, using the *previous* layer's embeddings:

```text
new_node[v] = MLP1(concat(node[v], sum_neighbor_nodes, sum_incident_edges))
new_edge[u,v] = MLP2(concat(node[u], node[v], edge[u,v]))
```

Each MLP has a hidden linear layer followed by ReLU and a linear output layer. Node and edge MLP parameters are separate. The default is three message-passing layers with width 16. No LayerNorm or residual connection is used.

The printed edge concatenation is ordered, but the graph is undirected. We average the edge MLP evaluated in both endpoint orders, and similarly average the two ordered final readouts. This explicit implementation choice enforces endpoint-swap invariance and node-permutation equivariance before deterministic tie-breaking. A unit test checks it.

After the final layer, concatenate both endpoint node embeddings and the edge embedding, then apply a bias-free linear readout. A sigmoid converts its logit to `p = probability(anti-alignment)`. Thus `p<0.5` chooses merge and `p>=0.5` chooses flip-merge, matching Algorithm 1's threshold convention.

Mini-batches concatenate graphs into a disjoint union. Node indices are shifted so messages never cross graph boundaries; loss normalization is still performed per graph.

## 7. The hybrid supervised loss

For labeled edges of one graph, use

```text
p[i] = sigmoid(logit[i])
c[i] = softmax(abs(p[i]-0.5) / T)
w[i] = lambda*c[i] + (1-lambda)
loss = sum_i w[i] * BCE(p[i], label[i])
```

Neutral edges are excluded **before** the confidence softmax and loss. This resolves an unspecified detail of the paper; the masking and gradients are tested. All-neutral graphs contribute a differentiable zero loss.

The implementation follows the printed unnormalized uniform term `(1-lambda)`, not `(1-lambda)/N`. Therefore total weight is `lambda + (1-lambda)*N`, and graph size affects loss scale. Confidence weights are differentiable: gradients flow through `c`, rather than silently detaching it. BCE is computed from logits for numerical stability.

A batch averages these per-graph losses. The default uses `lambda=0.5`, temperature `T=0.1`, Adam learning rate `0.001`, batch size 32, global gradient clipping at 10, up to 100 epochs, and early stopping after 20 non-improving validation epochs. The lowest validation objective selects the checkpoint. Temperature, batch size, clipping, patience, and width 16 are reconstruction choices; the paper does not fully specify them.

The eight ablations use seed 0 and the same data:

- BCE (`lambda=0`), MSE (sum of squared probability error), and softmax-only BCE (`lambda=1`).
- One, two, four, and five message-passing layers.
- Hidden width 2 with the default three-layer hybrid model.

Each ablation selects its own checkpoint using its own validation loss; their raw loss magnitudes are not directly comparable. Accuracy and downstream compression quality provide common comparisons.

## 8. Important paper ambiguities and our choices

### Confidence versus entropy in Algorithm 1

The prose says to choose the most confident edge. However, Algorithm 1 defines

```text
C = -p*log(p) - (1-p)*log(1-p)
```

and selects **maximum** `C`. That is maximum Bernoulli entropy: the least confident prediction, near `p=0.5`.

We implement both:

- `method="granite"`: maximize certainty, following the prose; equivalently minimize entropy.
- `method="paper_entropy"`: maximize entropy, following the printed algorithm literally.

For numerical stability, rank absolute logits rather than saturated float32 probabilities: larger `abs(logit)` means more confident; smaller means higher entropy. Both modes are evaluated with the **same trained models**. This comparison is reported rather than silently choosing an interpretation and calling it exact reproduction.

### Hidden dimensions and features

The method lists three node features and two edge features. The hyperparameter paragraph later says the feature sizes are “2 and 2” and describes the MLP sizes only loosely. We use the full three-feature node list with width 16 and include a width-2 ablation. Neither is asserted to recover an undocumented author implementation.

### Reduction ratio

Algorithm 1 stops when the remaining size reaches a fraction `alpha` of the original size. Some experimental prose calls reduction an edge percentage; another metric uses physical qubits. These are different quantities.

Here `reduction=r` means remove up to the requested fraction of **logical augmented vertices**, with target `ceil((1-r)*initial_vertices)`. The actual achieved fraction and remaining edge count are recorded separately. For example, 75% requested reduction of 14 spins leaves 4, which is actually 71.4% removal. No minor embedding is performed, so we do not call these physical-qubit savings.

### Optimum preservation

Learned labels are predictions, not persistency certificates. An incorrect merge can lose the ground state. The exact small-instance experiments measure that loss directly, including the accumulated offset and full reconstruction.

## 9. Datasets and experiment design

The desktop training dataset has 900 graphs: five sizes (4, 6, 8, 10, 12), three requested density settings, three topology families, and 20 instances per configuration. Each configuration contributes 16 training and 4 validation graphs: **720 training, 180 validation**. All fields are zero; weights are continuous uniform samples in [−5,5]. Models train on the original random graphs, not augmented intermediate compression trajectories.

| Generator | Parameterization |
|---|---|
| ER | `p = requested_degree/(n-1)` |
| BA | attach to `round(requested_degree/2)` old vertices, clipped to valid bounds |
| WS | nearest implemented even ring degree not exceeding `n-1`, rewiring probability 0.3 |

Requested degrees are 2, `max(2,n//2)`, and `n-1`. These controls are **not equal exact mean degrees across families**: BA finite-size degrees differ, and WS requires an even degree. Some settings coincide for small sizes. Actual average degrees and every generator seed are saved in `results/data/metadata.json`. All graphs have independent coefficient samples even when their topologies coincide.

The paper reports 97,500 graphs over sizes 2–26 and a separate 20,000-graph evaluation set. We do not claim those datasets match this reconstruction. Exact enumeration scales exponentially, so the included configuration deliberately uses smaller labeled graphs.

### Exact small-instance study

- Held-out sizes 14 and 16, larger than every training graph.
- ER, BA and WS; requested degree 4; three instances per size/topology: 18 graphs.
- Compression requests 0%, 12.5%, 25%, 50%, 75%.
- Exact solutions of both the original and every residual problem isolate compression error from solver error.
- Main models, entropy interpretation, untrained controls, random contraction, greedy contraction, and all eight ablations are evaluated.

The paper's optimality metric is reported as a percentage:

```text
100 * (1 - abs(E_reconstructed - E_optimum)/abs(E_optimum)).
```

It is not an approximation guarantee and can be negative. It is left undefined when the exact optimum is zero. We also report the energy gap and whether the exact optimum is preserved, which are easier to interpret.

### Larger classical study

- Sizes 25, 50, 100, 200 and 400; three families; two instances per size/topology: 30 graphs.
- The same five compression requests.
- Three trained GRANITE seeds, random contraction seed 0, and deterministic greedy contraction.
- Residual solving uses 16 Metropolis chains with 40 sweeps, inverse temperature geometrically increasing from 0.05 to 5, followed by best-improving single-spin flips on the best state seen.

The reference energy is a seeded simulated-annealing run on the **uncompressed** instance. It is not a proven optimum. A compressed run can outperform this reference; that does not violate the fact that contraction restricts the feasible set, because the reference solver is approximate. We report `(energy-reference)/sum(abs(edge_weights))`, not an “optimality” percentage, for these larger cases.

All methods use the same residual solver seed for a given original instance. The same reads/sweeps across sizes are not equal computational work. Compression and residual-solve times are recorded separately, and total pipeline time includes both. Offline GNN training and model loading are excluded. Exact-study time includes exhaustive residual solving; it is not comparable to QPU annealing latency.

### Baselines and statistics

`greedy_weight` satisfies the largest-magnitude current edge. `random` chooses a random current edge and random relationship. `untrained` uses the saved initial network. All methods use the same contraction algebra and reconstruction, so differences arise from decisions rather than different energy conventions.

For compression summaries, model seeds are averaged within each graph before averaging graphs. Error bars are one standard error across graphs, not 95% confidence intervals or significance tests. Pooled edge-classification accuracy is explicitly labeled. Ablation comparisons use seed 0 for both the baseline and controls. The small sample does not establish the best architecture or universal superiority.

## 10. What the experiments found

See the [complete report](results/evaluation/REPORT.md), including results that disagree with the paper's qualitative conclusions.

- The three trained models achieved **80.15% pooled edge accuracy** on the held-out 14/16-spin graphs, versus 49.27% for their untrained initializations.
- At 75% **requested logical** reduction, GRANITE's mean exact optimality metric was **99.80%**. The original optimum survived in **50 of 54** graph/model runs. This is a small classical test, not the paper's QPU result.
- The literal maximum-entropy selection achieved only **67.51%** mean optimality at that reduction, providing a concrete demonstration of why the algorithm/prose discrepancy matters.
- Greedy weight contraction remained strong: **99.26%** mean exact optimality at 75% requested reduction. At 200 and 400 spins, this simple baseline outperformed the small trained model relative to the common annealing reference.
- The small ablations did **not** establish three layers or hybrid loss as uniquely best; some other depths or losses performed better on this sample. Their results are retained without selecting a new default based on the test set.

The strongest supported conclusion is that the implemented classifier can guide useful partial compression on these small synthetic problems, while size transfer and advantage over simple heuristics remain limited.

## 11. Export for another solver

```bash
python -m granite.demo --export runs/demo-residual.json
```

The export contains linear coefficients, quadratic edges, the offset, residual vertex order, contraction trace, and auxiliary-node information. It uses the conventional **positive-sign** BQM energy `sum J*s*s + sum h*s + offset`, so it negates this package's `J` and `h`. This conversion is covered by an exhaustive small test.

A caller can construct a `dimod.BinaryQuadraticModel` from those coefficients, sample it with a chosen solver, and use the residual vertex order and trace to reconstruct spins. The included code does not submit QPU jobs, acquire embeddings, or report hypothetical physical-qubit counts. A future hardware study must separately measure embedding, physical qubits, chain strengths, chain-break handling, solver parameters and QPU timing.

## 12. Larger budgets and student exercises

`configs/paper_hyperparameter_attempt.json` requests 300 epochs and hidden width 2, while keeping other explicit reconstruction choices. It is **not** an exact paper configuration and was not run as the default study. It can be used with a separately generated larger dataset. Label generation has a guard at 18 spins; reproducing 26-spin exhaustive labeling requires an intentionally revised and benchmarked labeling strategy, not simply changing a config name.

Suggested exercises:

1. Derive the offset update by substituting `s[v]=d*s[u]` into the energy.
2. Enumerate the six ground states of an antiferromagnetic triangle and explain why all edges are neutral.
3. Plot Bernoulli entropy and explain the two inference modes.
4. Derive the loss gradient through the confidence softmax and compare with a detached-weight implementation.
5. Compare edge accuracy with the fraction of instances retaining the optimum.
6. Change graph density or introduce fields, then evaluate without selecting models on the test set.
7. Compare the printed hybrid weighting with a genuinely normalized uniform mixture; explain the graph-size scaling difference.
8. Add a trusted residual solver or actual hardware experiment and distinguish logical reduction from embedding-dependent physical reduction.

The tests include exact restricted-energy identities, auxiliary-spin reconstruction, neutral labels, cancellation and isolated nodes, hybrid-loss finite differences, endpoint/node permutation symmetry, disjoint graph batching, training/checkpoint smoke tests, solver reproducibility, and export sign conversion. Passing them verifies implementation properties; it does not prove general optimization performance.

## Citation and provenance

```bibtex
@inproceedings{tran2025granite,
  title={Scalable Quantum-Inspired Optimization Through Dynamic Qubit Compression},
  author={Tran, Co and Tran, Quoc-Bao and Son, Hy Truong and Dinh, Thang N.},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={11},
  pages={11353--11361},
  year={2025},
  doi={10.1609/aaai.v39i11.33235}
}
```

The user-supplied published PDF is the primary specification. Its SHA-256 and the execution environment are recorded in `results/environment.json`; the PDF is not redistributed. File checksums are in `results/SHA256SUMS.json`, and evaluated checkpoint hashes are in `results/evaluation/protocol.json`.
