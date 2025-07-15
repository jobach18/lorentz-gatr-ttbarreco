# Top Reconstruction with Standard Transformer

This experiment implements top quark reconstruction using a standard transformer architecture as a comparison baseline to the GATr-based `top_reco` experiment.

## Key Differences from `top_reco`

1. **Architecture**: Uses a standard transformer instead of GATr (no geometric algebra)
2. **Batching**: Uses standard PyTorch batching with padding instead of torch_geometric batching
3. **Attention**: Standard scaled dot-product attention instead of geometric attention
4. **Input Processing**: Direct 4-momentum vectors without geometric algebra embedding

## Files

- `dataset.py`: Dataset class that loads data and handles standard batching with padding
- `wrappers.py`: Model wrapper that combines transformer with output prediction layers
- `experiment.py`: Main experiment class that handles training and evaluation
- `../config/model/transformer_reco.yaml`: Model configuration
- `../config/top_reco_transformer.yaml`: Experiment configuration

## Usage

```bash
python run.py --config-name=top_reco_transformer
```

## Model Architecture

1. **Input**: Variable-length sequences of jet 4-momenta (px, py, pz, E)
2. **Transformer**: Standard multi-head self-attention with positional encoding
3. **Aggregation**: Global average pooling over valid (non-padded) jets
4. **Output**: MLP prediction of 2 top quark 4-momenta

## Configuration

- **Batch size**: 64 (smaller than GATr due to quadratic attention complexity)
- **Model size**: 128 hidden dimensions, 6 transformer blocks, 8 attention heads
- **Learning rate**: 1e-4 with Adam optimizer
- **Data scaling**: Minkowski normalization (same as original experiment)

## Expected Performance

This serves as a baseline comparison to evaluate the benefits of geometric algebra and GATr for physics problems. The standard transformer should perform reasonably but may be less sample-efficient than GATr for this geometric problem.
