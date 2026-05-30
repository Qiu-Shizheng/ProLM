# ProLM

<p align="center">
  <b>A transformer model for proteomics-based disease prediction</b>
</p>

<p align="center">
  <a href="#installation">Installation</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#input-files">Input Files</a> |
  <a href="#pretrained-checkpoint">Pretrained Checkpoint</a> |
  <a href="#proteinbert-embeddings">ProteinBERT Embeddings</a>
</p>

---

## Overview

ProLM represents each protein with both its measured expression value and an external sequence-derived embedding from ProteinBERT. The protein tokens are passed through a transformer encoder and fine-tuned for binary disease prediction.

The model can run in two attention modes:

- **Standard self-attention**, when no protein correlation matrix is supplied.
- **Correlation-biased self-attention**, when an optional protein correlation matrix is supplied.

## Installation

```bash
git clone https://github.com/Qiu-Shizheng/ProLM.git
cd ProLM
python -m pip install -e .
```

## Quick Start

Create example inputs:

```bash
python scripts/generate_synthetic_data.py --output-dir examples/synthetic
```

Fine-tune a small model:

```bash
python -m prolm.finetune \
  --expression-csv examples/synthetic/expression.csv \
  --labels-csv examples/synthetic/labels.csv \
  --protein-embeddings examples/synthetic/protein_embeddings.npz \
  --output-dir outputs/example_run \
  --epochs 2 \
  --batch-size 4 \
  --hidden-size 32 \
  --num-layers 1 \
  --chunk-size 16
```

Output files:

```text
outputs/example_run/
  best_classifier.pt
  metrics.json
  protein_order.txt
```

## Input Files

### Protein Expression

The expression table must contain an `eid` column and one column per protein. Protein columns are used in the order provided by the file.

```csv
eid,a1bg,aamdc,aarsd1,abca2,abhd14b
1,0.14,-1.20,0.73,0.44,-0.31
2,-0.35,0.41,1.08,-0.10,0.62
3,1.22,0.03,-0.64,0.51,-0.79
```

### Disease Labels

Labels are binary by default: `0` for control and `1` for case.

```csv
eid,label
1,0
2,1
3,0
```

### ProteinBERT Sequence Embeddings

The sequence embedding file is an `.npz` archive. Each key must match a protein column name in the expression file.

```python
import numpy as np

np.savez(
    "protein_embeddings.npz",
    a1bg=np.random.normal(size=15599).astype("float32"),
    aamdc=np.random.normal(size=15599).astype("float32"),
    aarsd1=np.random.normal(size=15599).astype("float32"),
)
```

### Optional Protein Correlation Matrix

If supplied, the correlation matrix must use protein names as both row names and column names.

```csv
,a1bg,aamdc,aarsd1
a1bg,1.00,0.12,-0.04
aamdc,0.12,1.00,0.20
aarsd1,-0.04,0.20,1.00
```

Run with correlation-biased attention:

```bash
python -m prolm.finetune \
  --expression-csv examples/synthetic/expression.csv \
  --labels-csv examples/synthetic/labels.csv \
  --protein-embeddings examples/synthetic/protein_embeddings.npz \
  --correlation-csv examples/synthetic/protein_correlation.csv \
  --output-dir outputs/example_with_correlation
```

## Pretrained Checkpoint

Fine-tune from a pretrained ProLM checkpoint:

```bash
python -m prolm.finetune \
  --expression-csv path/to/expression.csv \
  --labels-csv path/to/labels.csv \
  --protein-embeddings path/to/protein_embeddings.npz \
  --pretrained-checkpoint path/to/prolm_pretrained.pt \
  --output-dir outputs/my_task \
  --unfreeze-last-n-layers 2 \
  --epochs 20 \
  --batch-size 4 \
  --learning-rate 5e-5
```

Layer freezing:

| Option | Behavior |
|---:|---|
| `--unfreeze-last-n-layers 0` | train only the classifier head |
| `--unfreeze-last-n-layers 1` | fine-tune the final transformer layer |
| `--unfreeze-last-n-layers 2` | fine-tune the final two transformer layers |
| `--unfreeze-last-n-layers 24` | fine-tune all layers for a 24-layer checkpoint |

Checkpoint tensors are loaded when their names and shapes match the current model. This lets the same checkpoint be reused with a different protein panel as long as the ProteinBERT embedding dimension is compatible.

## ProteinBERT Embeddings

Prepare a sequence table:

```csv
Protein,Sequence
a1bg,MSMLVVFLLLWGVTWGPVTEAAIFYETQPSLWAE...
aamdc,MTSPEIASLSWGQMKVKGSNTTYKDCKVWPGGS...
aarsd1,MAFWCQRDSYAREFTTTVVSCCPAELQTEGSNG...
```

Extract ProteinBERT global embeddings:

```bash
python scripts/extract_proteinbert_embeddings.py \
  --sequences-csv path/to/sequences.csv \
  --output-npz path/to/protein_embeddings.npz \
  --excluded-output outputs/excluded_long_sequences.txt \
  --proteinbert-dir path/to/proteinbert_model_dir \
  --proteinbert-checkpoint full_go_epoch_92400_sample_23500000.pkl \
  --batch-size 64
```

The resulting `.npz` file can be passed to `--protein-embeddings`.

## GPU and Memory

Fine-tuning uses CUDA by default. To run on CPU:

```bash
python -m prolm.finetune ... --device cpu
```

Approximate memory use depends mainly on the checkpoint size, number of proteins, chunk size, and number of unfrozen layers. For a 24-layer, hidden-size 768 checkpoint with about 2,900 proteins and chunk size 512:

| Batch size | Unfrozen layers | Approximate GPU memory |
|---:|---:|---:|
| 1 | 0 | 3-5 GB |
| 2 | 0 | 5-8 GB |
| 4 | 0 | 8-12 GB |
| 4 | 2 | 12-18 GB |

For smaller GPUs, start with `--batch-size 1` and `--unfreeze-last-n-layers 0`, then increase gradually.

## Python API

```python
from prolm import load_prolm_data
from prolm.model import ProLMBackbone, ProLMClassifier, freeze_backbone

data = load_prolm_data(
    expression_csv="expression.csv",
    labels_csv="labels.csv",
    protein_embeddings_npz="protein_embeddings.npz",
)

backbone = ProLMBackbone(
    feature_dim=data.features_tensor.shape[1],
    hidden_size=768,
    num_layers=24,
    num_proteins=len(data.protein_names),
    features_tensor=data.features_tensor,
    bias_matrix=data.bias_matrix,
)
freeze_backbone(backbone, unfreeze_last_n_layers=2)
model = ProLMClassifier(backbone, num_classes=2)
```
