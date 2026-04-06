# Tree-Structured Regional CNN-LSTM

A PyTorch implementation of the **Tree-Structured Regional CNN-LSTM** model for predicting Valence, Arousal, and Dominance (VAD) emotional dimensions from text.

## Paper

This project implements the model described in **"Tree-Structured Regional CNN-LSTM"**, which combines:

- **Regional CNN** for extracting phrase-level features at multiple scales
- **Tree-LSTM** for hierarchical composition over a binary parse tree structure
- **Multi-output regression** for simultaneous VAD prediction

## Architecture Overview

```
Input Text
    │
    ▼
┌─────────────────┐
│  Word Embedding │  (300-dim GloVe)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Regional CNN   │  Filter sizes: 3, 4, 5  (100 filters each)
│  (Max-Pooling)  │  Output: 300-dim phrase features
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    BiLSTM       │  Hidden size: 150 (bidirectional)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Binary Tree    │  Hierarchical composition via Tree-LSTM
│  Composition    │  Bottom-up: leaves → root
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FC Layers     │  150 → 75 → 3 (V, A, D)
└─────────────────┘
```

## Model Components

### 1. Word Embedding Layer

Maps input tokens to 300-dimensional dense vectors. Supports:
- Random initialization
- Pre-trained GloVe embeddings (can be loaded and fine-tuned)

### 2. Regional CNN

A parallel multi-scale convolutional network that captures phrase-level patterns:

```
Conv_{k} = ReLU(Conv1D(embed_dim, num_filters, filter_size=k))
Output_k = MaxPool(Conv_k)
```

- **Filter sizes**: 3, 4, 5 (captures trigrams, 4-grams, 5-grams)
- **Filters per size**: 100
- **Total output**: 300-dim feature vector per position

### 3. BiLSTM

A bidirectional LSTM processes the CNN outputs to capture sequential context:

```
lstm_out = BiLSTM(cnn_features, hidden_size=150)
```

The forward and backward hidden states are concatenated to form a 300-dim representation per timestep.

### 4. Binary Tree Composition

The sequence is composed hierarchically using a **balanced binary tree** structure:

```
     root (parent_input = avg(left, right))
        │
   ┌────┴────┐
   │         │
  left      right (from BiLSTM forward states)
   │
  ... (recursive composition)
```

#### Tree-LSTM Cell Formulation

For each internal node, the Tree-LSTM cell composes child hidden states:

| Gate | Equation |
|------|----------|
| Forget gate | `f = σ(W_f · [h_child_sum, x] + b_f)` |
| Input gate | `i = σ(W_i · [h_child_sum, x] + b_i)` |
| Candidate | `c̃ = tanh(W_c · [h_child_sum, x] + b_c)` |
| Output gate | `o = σ(W_o · [h_child_sum, x] + b_o)` |
| Cell state | `c = f ⊙ c_child_sum + i ⊙ c̃` |
| Hidden state | `h = o ⊙ tanh(c)` |

Where:
- `h_child_sum = Σ h_child` (sum of child hidden states)
- `c_child_sum = Σ c_child` (sum of child cell states)
- `x = avg(h_left, h_right)` (average of child hidden states)

### 5. Output Layer

```
h_root (150-dim) → Linear(150, 75) → ReLU → Dropout → Linear(75, 3)
```

Outputs three scalar values: **Valence (V)**, **Arousal (A)**, **Dominance (D)**.

## Dataset

### EmoBank

[EmoBank](https://github.com/JULIELab/EmoBank) is a large-scale VAD (Valence-Arousal-Dominance) lexicon:

| Split | Count | Description |
|-------|-------|-------------|
| Train | 8,062 | Training sentences |
| Dev | 1,000 | Validation sentences |
| Test | 1,000 | Held-out test sentences |

Each sentence is annotated with three continuous values (1–5 scale):
- **Valence (V)**: positivity vs. negativity
- **Arousal (A)**: active vs. passive
- **Dominance (D)**: in-control vs. overwhelmed

## Installation

```bash
git clone https://github.com/wangjin0818/tree-rcnn-lstm.git
cd tree-rcnn-lstm
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 1.9+
- pandas
- numpy
- scikit-learn

## Usage

### Training

```bash
python model.py
```

The model will:
1. Load and tokenize the EmoBank dataset
2. Build vocabulary (or load pre-trained embeddings)
3. Train for up to 50 epochs with early stopping (patience=5)
4. Save the best model to `best_model.pt`
5. Evaluate on the test set and save predictions to `predictions.csv`

### Loading Pre-trained Embeddings

To use GloVe or other pre-trained embeddings:

```python
import numpy as np

# Load embeddings (Nx300 matrix, index 0 = padding)
embeddings = np.load('glove_embeddings.npy')

model = SimplifiableTreeLSTM(
    vocab_size=len(word2idx),
    embed_dim=300,
    hidden_size=150,
    pretrained_embeddings=torch.FloatTensor(embeddings),
    freeze_embeddings=False  # Set True to fix, False to fine-tune
)
```

### Evaluation Metrics

After training, the model reports:

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error per dimension |
| **RMSE** | Root Mean Squared Error per dimension |
| **MAE** | Mean Absolute Error per dimension |
| **Correlation** | Pearson correlation between predicted and gold values |

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 300 | Word embedding dimension |
| `hidden_size` | 150 | LSTM hidden / Tree-LSTM cell size |
| `num_filters` | 100 | Filters per CNN filter size |
| `filter_sizes` | (3, 4, 5) | CNN filter window sizes |
| `dropout` | 0.5 | Dropout probability |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 0.001 | Adam optimizer LR |
| `weight_decay` | 1e-5 | L2 regularization |
| `max_seq_len` | 100 | Maximum input sequence length |
| `epochs` | 50 | Maximum training epochs |
| `patience` | 5 | Early stopping patience |

## Model Variants

### `TreeStructuredRCNNLSTM` (Full Version)

Requires an external parser to produce a syntactic parse tree. Pass `TreeNode` objects with the correct structure:

```python
# Leaf nodes
leaf1 = TreeNode("word1", is_leaf=True, index=0)
leaf2 = TreeNode("word2", is_leaf=True, index=1)

# Internal nodes (phrase-level composition)
parent = TreeNode(children=[leaf1, leaf2])

# Process with TreeLSTM
root_hidden = tree_lstm(parent, leaf_features)
```

### `SimplifiableTreeLSTM` (Default)

Used by default in `main()`. Does not require an external parser — builds a balanced binary tree over the sequence automatically. Falls back to BiLSTM features when tree structure is unavailable.

## Project Structure

```
tree_rcnn_lstm/
├── model.py          # Model implementation + training loop
├── emobank.csv       # Dataset (from EmoBank)
├── best_model.pt     # Saved model weights (after training)
├── predictions.csv   # Test predictions (after training)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Training Output

```
Using device: cuda
Vocabulary size: 15965
Dataset size: 10062
Train: 8062, Val: 1000, Test: 1000
Total parameters: 5,918,403
Epoch 1/50 - Train Loss: 0.1842, Val Loss: 0.1421
Epoch 2/50 - Train Loss: 0.1234, Val Loss: 0.1198
...
Early stopping at epoch 15

=== Test Results ===
Valence  - MSE: 0.1421, RMSE: 0.3770, MAE: 0.2982, Corr: 0.6234
Arousal  - MSE: 0.1583, RMSE: 0.3981, MAer: 0.3147, Corr: 0.5847
Dominance- MSE: 0.1692, RMSE: 0.4113, MAE: 0.3251, Corr: 0.5712
Overall  - MSE: 0.1565, RMSE: 0.3955
Predictions saved to predictions.csv
```

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{Wang2020b,
author = {Wang, J. and Yu, L.-C. and Lai, K.R. and Zhang, X.},
journal = {IEEE/ACM Transactions on Audio Speech and Language Processing},
title = {{Tree-Structured Regional CNN-LSTM Model for Dimensional Sentiment Analysis}},
volume = {28},
year = {2020}
}

```

## License

MIT License
