# File formats

All files are NumPy `.npz` archives. Mode order is always **B1, T1, B2, T2**
(first bending, first torsion, second bending, second torsion).

## Population sample — `sample_{id:04d}_Ng{g}.npz`

Written by `scripts/generate_population.py`, one file per deck. *N* graph nodes,
*E* directed edges.

| key | shape, dtype | meaning |
|---|---|---|
| `sample_id`, `Ng`, `Nx` | scalar int64 | deck id, girder count, number of girder bays |
| `params_json` | str | the sampled parameters (`population.sample_params`) |
| `graph_node_ids` | (N,) int64 | FE node id of each graph node (1-based) |
| `coords_xy_mm` | (N, 2) float64 | plan coordinates; x along the span from the hinge, y across |
| `edge_index` | (2, E) int64 | directed edges: undirected edges (i < j), then the same reversed |
| `edge_length_mm`, `edge_theta_rad` | (E,) float64 | plan length and angle of each directed edge |
| `freq_bins_hz` | (400,) float64 | frequencies of the spectral features, k × 0.09765625 Hz |
| `x_psd` | (N, 400) float32 | **node input**: Welch PSD of vertical acceleration, divided by its maximum over the full one-sided spectrum |
| `edge_attr_cpsd_full` | (E, 800) float32 | **edge input**: complex coherency, Re and Im interleaved per bin |
| `y_mode_shape_signed` | (N, 4) float64 | **target**: vertical mode shapes, sign-aligned to the templates, divided by their max \|value\| over all FE nodes |
| `y_freq_hz` | (4,) float32 | **target**: natural frequencies |
| `matched_mode_indices` | (4,) int64 | which of the 15 FE modes each target is (0-based) |
| `matched_mac` | (4,) float64 | MAC of each against its template (all ≥ 0.85) |
| `target_mode_names` | (4,) str | `B1 T1 B2 T2` |

The generator also writes `summary.csv` (one row per deck: parameters,
frequencies, MAC, FE mode index) and `failures.log` (every rejected draw and why).

## Measured deck — input of `predict_measured.py` / `finetune_loocv.py`

Read by `linkspan_gnn.measured.load_measured`, written by `save_measured`.

| key | shape | meaning |
|---|---|---|
| `name` | str | deck name (used in output file names) |
| `fe_params_json` | str | JSON dict of FE parameters overriding `fe_model.default_params()` — at least `L_mm`, `Nx`, `Ng`, `dy_mm`, `diag_pattern`, `bottom_beam`, `balcony`, `nsm` (see the example below) |
| `sensor_xy_mm` | (S, 2) | plan position of each vertical accelerometer, in the FE frame: x from the hinge (x = 0) towards the free end, y across with girder 0 at y = 0 |
| `acc` | (S, T) | vertical acceleration, one row per sensor, in the same order as `sensor_xy_mm` (any unit: the features are normalised) |
| `fs_hz` | scalar | sampling rate (the spectra are interpolated onto the 200 Hz training grid, so any rate above 40 Hz works) |
| `ref_mode_shapes` | (S, 4), optional | reference mode shapes at the sensors (e.g. from SSI); needed for scoring and fine-tuning. Sign and scale do not matter for MAC; for fine-tuning they should be scaled like the training targets (max \|value\| = 1) |
| `ref_freq_hz` | (4,), optional | reference natural frequencies; needed for fine-tuning |

An example geometry, in the units of `fe_model` (mm, tonne):

```python
fe_params = {
    "L_mm": 18000, "Nx": 29, "Ng": 4, "dy_mm": 3000, "diag_pattern": "None",
    "bottom_beam": {"x_frac": 0.667, "overhang_mm": 600},
    "balcony": {"enable": True, "side": "both", "x_center_frac": 0.86, "nBeams": 3,
                "dx_mm": 675, "length_mm": 2500.0, "edge_beam_enable": True, "brace_enable": True},
    "nsm": {"enable": True, "total_mass_tonne": 30},
}
```

Only the geometry matters here: the FE model of a measured deck is built to
obtain its graph (nodes and members in plan), so material, mass and support
stiffness have no effect. `examples/make_synthetic_measured.py` writes complete
example files.

Each sensor becomes an extra graph node, placed on the girder edge it lies on
(see `measured.measured_graph`); sensors far from any girder still work but are
joined to their nearest FE node only.
