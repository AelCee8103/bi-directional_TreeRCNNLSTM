"""
Tree-Structured Regional CNN-LSTM for VAD (Valence-Arousal-Dominance) Prediction
Based on the paper: "Tree-Structured Regional CNN-LSTM"

Architecture:
1. Word Embeddings (GloVe 300-dim)
2. Regional CNN for phrase-level feature extraction
3. Tree-LSTM for hierarchical composition
4. Regression output layer for V, A, D prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import defaultdict
import re
import os
import spacy
import benepar
from nltk.tree import Tree

# Initialize global parser
nlp = spacy.load('en_core_web_md')
if spacy.__version__.startswith('2'):
    nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

def build_syntax_tree(text, max_len=100):
    """Parses text and returns the root TreeNode and a mapping of leaf node indices."""
    if not text or str(text).strip() == "":
        return TreeNode("empty", is_leaf=True, index=0), [0]

    doc = nlp(str(text))
    if not list(doc.sents):
        return TreeNode("empty", is_leaf=True, index=0), [0]
    
    # Get NLTK tree for the first sentence 
    # (For multi-sentence text, you'd combine them, but we'll use sentence 1 for simplicity)
    sent = list(doc.sents)[0]
    try:
        nltk_tree = Tree.fromstring(sent._.parse_string)
    except Exception:
        return TreeNode("error", is_leaf=True, index=0), [0]

    leaf_indices = []
    
    def convert_nltk_to_treenode(nltk_node):
        if type(nltk_node) == str:  
            # It's a leaf/word
            idx = len(leaf_indices)
            if idx >= max_len:  # Cap at max_seq_len
                idx = max_len - 1
            leaf_indices.append(idx)
            return TreeNode(nltk_node, is_leaf=True, index=idx)
            
        # It's an internal phrase node
        children = [convert_nltk_to_treenode(child) for child in nltk_node]
        return TreeNode(nltk_node.label(), children=children, is_leaf=False)

    root_node = convert_nltk_to_treenode(nltk_tree)
    return root_node, leaf_indices

class TreeNode:
    """Represents a node in the parse tree."""
    def __init__(self, text, children=None, is_leaf=False, index=None):
        self.text = text
        self.children = children if children else []
        self.is_leaf = is_leaf
        self.hidden = None
        self.index = index  # For leaf nodes, the word index


class RegionalCNN(nn.Module):
    """
    Regional CNN extracts phrase-level features from word embeddings.
    Uses multiple filter sizes to capture different n-gram patterns.
    """
    def __init__(self, embed_dim, num_filters=100, filter_sizes=(3, 4, 5)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs)
            for fs in filter_sizes
        ])
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes

    def forward(self, x, mask=None):
        # x: (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)

        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch, num_filters, seq_len - fs + 1)
            pooled = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch, num_filters)
            pooled_outputs.append(pooled)

        return torch.cat(pooled_outputs, dim=1)  # (batch, num_filters * len(filter_sizes))


class TreeLSTMCell(nn.Module):
    """
    Tree-LSTM Cell as described in the paper.
    Composes child node hidden states with the input.

    Equations:
    - f = σ(W_f · [h_{j}^{l-1}, x_{j}^{l}] + b_f)  # forget gate
    - i = σ(W_i · [h_{j}^{l-1}, x_{j}^{l}] + b_i)  # input gate
    - c̃ = tanh(W_c · [h_{j}^{l-1}, x_{j}^{l}] + b_c)  # candidate cell state
    - o = σ(W_o · [h_{j}^{l-1}, x_{j}^{l}] + b_o)  # output gate
    - c = f ⊙ c_{j}^{l-1} + i ⊙ c̃  # cell state
    - h = o ⊙ tanh(c)  # hidden state
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Combined transformation for efficiency: [h, x] -> gates
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    def forward(self, x, child_h, child_c):
        """
        Args:
            x: (batch, input_size) - input features (from CNN or child composition)
            child_h: list of (batch, hidden_size) - hidden states of children
            child_c: list of (batch, hidden_size) - cell states of children
        Returns:
            h: (batch, hidden_size) - new hidden state
            c: (batch, hidden_size) - new cell state
        """
        if len(child_h) == 0:
            # Leaf node case: use zero initial states
            batch_size = x.size(0)
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
            child_h_sum = torch.zeros(batch_size, self.hidden_size, device=x.device)
            child_c_sum = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            # Sum child hidden/cell states
            child_h_sum = sum(child_h)
            child_c_sum = sum(child_c)

        # Concatenate child states with input
        combined = torch.cat([child_h_sum, x], dim=1)

        # Compute gates
        gates = self.W(combined)
        gates = gates.chunk(4, dim=1)

        f = torch.sigmoid(gates[0])  # forget gate
        i = torch.sigmoid(gates[1])  # input gate
        c_tilde = torch.tanh(gates[2])  # candidate cell
        o = torch.sigmoid(gates[3])  # output gate

        # Cell state and hidden state
        c = f * child_c_sum + i * c_tilde
        h = o * torch.tanh(c)

        return h, c


class TreeLSTM(nn.Module):
    """
    Tree-LSTM processes a parse tree bottom-up.
    Leaf nodes get features from word embeddings or CNN.
    Internal nodes compose child hidden states.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = TreeLSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, tree, leaf_features):
        """
        Process tree bottom-up.
        Args:
            tree: TreeNode object
            leaf_features: (num_leaves, batch, feature_dim) or dict mapping leaf indices
        Returns:
            root_hidden: (batch, hidden_size)
        """
        def process_node(node):
            if node.is_leaf:
                # Leaf node: get features from embedding/CNN output
                # All leaves get the same projected feature vector.
                feat = leaf_features[0]
                h, c = self.cell(feat, [], [])
                node.hidden = h
                node.cell = c
                return h, c
            else:
                # Handle internal nodes that might not have children
                if not node.children:
                    # If an internal node has no children, return zero vectors
                    h = torch.zeros(1, self.hidden_size, device=leaf_features[0].device)
                    c = torch.zeros(1, self.hidden_size, device=leaf_features[0].device)
                    node.hidden = h
                    node.cell = c
                    return h, c

                # Internal node: compose children
                child_h = []
                child_c = []
                for child in node.children:
                    process_node(child)
                    child_h.append(child.hidden)
                    child_c.append(child.cell)

                # Average of child features as input (can also use concatenation)
                child_h_avg = torch.stack(child_h).mean(dim=0)
                child_c_sum = sum(child_c)

                # Use average child hidden as input to this node
                h, c = self.cell(child_h_avg, child_h, child_c_sum)
                node.hidden = h
                node.cell = c
                return h, c

        root_h, _ = process_node(tree)
        return root_h


class TreeStructuredRCNNLSTM(nn.Module):
    """
    Full model: Tree-Structured Regional CNN-LSTM for VAD prediction.

    1. Word embedding layer
    2. Regional CNN for phrase-level features
    3. Tree-LSTM for hierarchical composition
    4. Regression output for V, A, D
    """
    def __init__(
        self,
        vocab_size,
        embed_dim=300,
        hidden_size=150,
        num_filters=100,
        filter_sizes=(3, 4, 5),
        dropout=0.5,
        pretrained_embeddings=None,
        freeze_embeddings=True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # Regional CNN
        self.cnn = RegionalCNN(embed_dim, num_filters, filter_sizes)
        cnn_output_dim = num_filters * len(filter_sizes)

        # Project CNN output to hidden size for leaf nodes
        self.leaf_proj = nn.Linear(cnn_output_dim, hidden_size)

        # Tree-LSTM
        self.tree_lstm = TreeLSTM(hidden_size, hidden_size)

        # Output layer for VAD regression
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)  # V, A, D
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, trees, leaf_indices):
        batch_size, seq_len = x.size()
        
        # 1. Embeddings and CNN
        embedded = self.embedding(x)
        cnn_features = self.cnn(embedded) # (batch, cnn_output_dim)

        root_hiddens = []
        for i, tree in enumerate(trees):
            # Get CNN features for this sentence's specific leaf structure
            # Project leaf features to the Tree-LSTM's input size
            leaf_feats_proj = self.leaf_proj(cnn_features[i:i+1]) # (1, hidden_size)
            
            # The original code was trying to map sequence indices to a single CNN vector.
            # A more standard approach is to use the single vector for all leaves.
            # Let's create a list of this feature for each leaf.
            tree_leaf_features = [leaf_feats_proj for _ in leaf_indices[i]]
            
            if not tree_leaf_features: # Handle empty trees
                # Create a zero tensor if there are no leaves
                root_h = torch.zeros(1, self.hidden_size, device=x.device)
            else:
                root_h = self.tree_lstm(tree, tree_leaf_features)

            root_hiddens.append(root_h.squeeze(0))

        root_hidden = torch.stack(root_hiddens, dim=0)
        root_hidden = self.dropout(root_hidden)
        return self.fc(root_hidden)


class SimplifiableTreeLSTM(nn.Module):
    """
    Simplified Tree-LSTM variant that doesn't require parsing trees.
    Uses a binary tree structure built from sentence parsing.
    Falls back to BiLSTM if parsing is unavailable.
    """
    def __init__(
        self,
        vocab_size,
        embed_dim=300,
        hidden_size=150,
        num_filters=100,
        filter_sizes=(3, 4, 5),
        dropout=0.5,
        pretrained_embeddings=None,
        freeze_embeddings=True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        # Word embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # Regional CNN
        self.cnn = RegionalCNN(embed_dim, num_filters, filter_sizes)
        cnn_output_dim = num_filters * len(filter_sizes)

        # BiLSTM as fallback when tree structure unavailable
        self.bilstm = nn.LSTM(
            cnn_output_dim,
            hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Tree-LSTM cell for hierarchical composition
        self.tree_cell = TreeLSTMCell(hidden_size, hidden_size)

        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)
        )

        self.dropout = nn.Dropout(dropout)
        self.cnn_output_dim = cnn_output_dim

    def build_binary_tree(self, seq_len):
        """
        Build a balanced binary tree for sequence of given length.
        Returns list of (parent_idx, left_child, right_child) tuples in bottom-up order.
        Leaf nodes are 0..seq_len-1, internal nodes start from seq_len.
        """
        if seq_len <= 1:
            return []

        current_nodes = list(range(seq_len))
        edges = []
        next_node_id = seq_len

        while len(current_nodes) > 1:
            new_level = []
            i = 0
            while i < len(current_nodes):
                if i + 1 < len(current_nodes):
                    left = current_nodes[i]
                    right = current_nodes[i + 1]
                    edges.append((next_node_id, left, right))
                    new_level.append(next_node_id)
                    next_node_id += 1
                    i += 2
                else:
                    new_level.append(current_nodes[i])
                    i += 1
            current_nodes = new_level

        # edges is already bottom-up: first edges combine leaves, later edges combine internal nodes
        return edges

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len) - word indices
            mask: (batch, seq_len) - attention mask
        Returns:
            vad_pred: (batch, 3) - predicted V, A, D
        """
        batch_size, seq_len = x.size()

        # Word embeddings
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Regional CNN features
        cnn_features = self.cnn(embedded)  # (batch, cnn_output_dim)

        # Expand CNN features for each position
        cnn_expanded = cnn_features.unsqueeze(1).expand(batch_size, seq_len, self.cnn_output_dim)

        # BiLSTM for sequential context
        lstm_out, _ = self.bilstm(cnn_expanded)  # (batch, seq_len, hidden_size * 2)

        # Build binary tree and compose hierarchically
        edges = self.build_binary_tree(seq_len)

        # BiLSTM output is (batch, seq_len, hidden_size * 2)
        # Split into forward (hidden_size) and backward (hidden_size) parts
        lstm_fw = lstm_out[:, :, :self.hidden_size]   # (batch, seq_len, hidden_size)
        lstm_bw = lstm_out[:, :, self.hidden_size:]   # (batch, seq_len, hidden_size)

        # Initialize: node_h stores forward-only states (hidden_size) for tree composition
        node_h = [lstm_fw[:, i, :] for i in range(seq_len)]  # (batch, hidden_size)
        # node_bw stores backward states for context (not used in tree composition)
        node_bw = [lstm_bw[:, i, :] for i in range(seq_len)]
        node_c = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(seq_len)]

        # Process bottom-up: edges connect children to parent
        for parent, left, right in edges:
            child_h = [node_h[left], node_h[right]]  # each (batch, hidden_size)
            child_c = node_c[left] + node_c[right]     # (batch, hidden_size)

            # parent_input: average of forward states from children (hidden_size)
            parent_input = (node_h[left] + node_h[right]) / 2

            h, c = self.tree_cell(parent_input, child_h, child_c)
            node_h.append(h)
            node_c.append(c)

        # Root is the last added internal node
        root_h = node_h[-1]  # (batch, hidden_size)
        root_h = self.dropout(root_h)

        # Predict VAD
        vad_pred = self.fc(root_h)

        return vad_pred

def tree_collate_fn(batch):
    x_batch = torch.stack([item[0] for item in batch])
    vad_batch = torch.stack([item[1] for item in batch])
    trees_batch = [item[2] for item in batch]
    leaf_idxs_batch = [item[3] for item in batch]
    return x_batch, vad_batch, trees_batch, leaf_idxs_batch
class EmoBankDataset(Dataset):
    """Dataset for Emobank VAD prediction."""

    def __init__(self, csv_path, vocab=None, word2idx=None, max_seq_len=100):
        self.data = pd.read_csv(csv_path)
        self.max_seq_len = max_seq_len

        # Build vocabulary if not provided
        if word2idx is None:
            self.word2idx = {'<PAD>': 0, '<UNK>': 1}
            self._build_vocab()
        else:
            self.word2idx = word2idx

        self.vocab_size = len(self.word2idx)

    def _build_vocab(self):
        """Build vocabulary from training data."""
        word_counts = defaultdict(int)
        for text in self.data['text']:
            words = self._tokenize(text)
            for word in words:
                word_counts[word] += 1

        # Keep most frequent words
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        for word, _ in sorted_words[:30000]:  # Max 30k words
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)

    def _tokenize(self, text):
        """Simple tokenization."""
        if pd.isna(text):
            return []
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        vad = torch.tensor([row['V'], row['A'], row['D']], dtype=torch.float32)

        words = self._tokenize(text)
        indices = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]
        
        # Build Syntactic Tree
        tree, leaf_idxs = build_syntax_tree(text, self.max_seq_len)

        if len(indices) > self.max_seq_len:
            indices = indices[:self.max_seq_len]
        else:
            indices = indices + [self.word2idx['<PAD>']] * (self.max_seq_len - len(indices))

        x = torch.tensor(indices, dtype=torch.long)

        # PyTorch dataloader needs list/tree structures to be properly batched,
        # so returning customs objects via dataloader often requires a custom `collate_fn`.
        return x, vad, tree, leaf_idxs


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=50,
    lr=0.001,
    weight_decay=1e-5,
    patience=5
):
    """Train the model with MSE loss for VAD regression."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x, vad, trees, leaf_indices = batch
            x = x.to(device)
            vad = vad.to(device)

            optimizer.zero_grad()
            pred = model(x, trees, leaf_indices)
            loss = criterion(pred, vad)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
             x, vad, trees, leaf_indices = batch # <-- UNPACK TREES
             x = x.to(device)
             vad = vad.to(device)
             pred = model(x, trees, leaf_indices) # <-- PASS TREES
             loss = criterion(pred, vad)
             val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model


def main():
    # Configuration
    DATA_PATH = './emobank.csv'
    EMBED_DIM = 300
    HIDDEN_SIZE = 150
    NUM_FILTERS = 100
    FILTER_SIZES = (3, 4, 5)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    MAX_SEQ_LEN = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset = EmoBankDataset(DATA_PATH, max_seq_len=MAX_SEQ_LEN)
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Dataset size: {len(dataset)}")

    # Split by predefined split column
    train_indices = []
    val_indices = []
    test_indices = []

    for idx, row in enumerate(dataset.data.itertuples()):
        if row.split == 'train':
            train_indices.append(idx)
        elif row.split == 'dev':
            val_indices.append(idx)
        elif row.split == 'test':
            test_indices.append(idx)

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # Create data loaders
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=tree_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=tree_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=tree_collate_fn)

    # Create model
    model = TreeStructuredRCNNLSTM(
        vocab_size=dataset.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        num_filters=NUM_FILTERS,
        filter_sizes=FILTER_SIZES,
        dropout=0.5
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train
    model = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=EPOCHS,
        lr=LEARNING_RATE
    )

    # Evaluate on test set
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            x, vad, trees, leaf_indices = batch 
            x = x.to(device)
            pred = model(x, trees, leaf_indices) 
            predictions.append(pred.cpu().numpy())
            targets.append(vad.numpy())

    predictions = np.vstack(predictions)
    targets = np.vstack(targets)

    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets), axis=0)
    correlation = np.array([
        np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
        for i in range(3)
    ])

    print("\n=== Test Results ===")
    print(f"Valence  - MSE: {mse[0]:.4f}, RMSE: {rmse[0]:.4f}, MAE: {mae[0]:.4f}, Corr: {correlation[0]:.4f}")
    print(f"Arousal  - MSE: {mse[1]:.4f}, RMSE: {rmse[1]:.4f}, MAE: {mae[1]:.4f}, Corr: {correlation[1]:.4f}")
    print(f"Dominance- MSE: {mse[2]:.4f}, RMSE: {rmse[2]:.4f}, MAE: {mae[2]:.4f}, Corr: {correlation[2]:.4f}")
    print(f"Overall  - MSE: {mse.mean():.4f}, RMSE: {rmse.mean():.4f}")

    # Save predictions
    results_df = dataset.data.iloc[test_indices].copy()
    results_df['V_pred'] = predictions[:, 0]
    results_df['A_pred'] = predictions[:, 1]
    results_df['D_pred'] = predictions[:, 2]
    results_df.to_csv('predictions.csv', index=False)
    print("\nPredictions saved to predictions.csv")


if __name__ == '__main__':
    main()
