# Pretrained network: `edgegat_N2000.pt`

The paper's main network.

| | |
|---|---|
| architecture | `linkspan_gnn.gnn.EdgeGATModalGNN`: hidden size 32, 3 EdgeGATConv layers, 4 heads, dropout 0.1 — 55 912 parameters |
| inputs | node PSD (400 bins), edge complex coherency (800 values) |
| outputs | signed mode shapes of B1, T1, B2, T2 at every node; their natural frequencies |
| training data | N = 2000 decks of the synthetic population, 500 per girder count (1700 training / 300 validation) |
| training | 5000 epochs, seed 123, CPU, 2 threads; checkpoint with the lowest validation loss (epoch 591) |

Load it with

```python
from linkspan_gnn.gnn import load_checkpoint, load_sample, predict
model, scaler, info = load_checkpoint("pretrained/edgegat_N2000.pt")
graph, freq_true = load_sample("data/population/sample_1250_Ng2.npz")
mode_shapes, freq_hz = predict(model, scaler, graph)
```

## Test-set performance

The 500 test decks of the population (125 per girder count, never trained or
validated on), `scripts/evaluate.py`: MAC against the FE mode shape at every
node, and the frequency error.

| decks | B1 | T1 | B2 | T2 | mean MAC | frequency MAE |
|---|---|---|---|---|---|---|
| all 500 | 0.957 | 0.902 | 0.932 | 0.939 | 0.933 | 0.219 Hz |
| Ng = 2 | 0.949 | 0.894 | 0.849 | 0.902 | 0.898 | 0.353 Hz |
| Ng = 3 | 0.958 | 0.898 | 0.964 | 0.951 | 0.943 | 0.171 Hz |
| Ng = 4 | 0.970 | 0.921 | 0.961 | 0.959 | 0.953 | 0.158 Hz |
| Ng = 5 | 0.952 | 0.893 | 0.954 | 0.945 | 0.936 | 0.193 Hz |

A test set regenerated with `scripts/generate_population.py` gives the same
numbers to the precision shown.
