"""Graph neural network for the modal properties of ferry linkspan decks.

Modules, in pipeline order:
  fe_model       parametric 3-D beam FE model and modal solver
  mode_matching  labels B1, T1, B2, T2 among the solved modes
  excitation     ambient (multi-support white-noise) response by modal superposition
  graph          FE model -> graph on the deck plane
  features       Welch PSD node features, complex-coherency edge features, propagation
  population     synthetic population: one training sample per deck
  gnn            the EdgeGATConv network, datasets, checkpoints
  training       training loop
  measured       measured decks: sensor graph, features, zero-shot, fine-tuning
"""

__version__ = "1.0.0"
