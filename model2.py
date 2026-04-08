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
import benepar
import nltk

benepar.download('benepar_en3')
parser = benepar.Parser("benepar_en3")

def custom_collate_fn(batch):
    """Custom collate to handle TreeNode objects which can't be stacked into a tensor."""
    x = torch.stack([item[0] for item in batch])
    vad = torch.stack([item[1] for item in batch])
    trees = [item[2] for item in batch]
    leaf_indices = [item[3] for item in batch]
    return x, vad, trees, leaf_indices

def text_to_treenode(text, max_seq_len):
    """Parses text with benepar and converts it to the model's TreeNode format."""
    def nltk_to_custom(nltk_tree, leaf_idx_counter):
        if isinstance(nltk_tree, str):
            # It's a leaf node (word)
            node = TreeNode(nltk_tree, is_leaf=True)
            # Clamp index to max_seq_len to avoid tensor out-of-bounds due to tokenization differences
            node.index = min(leaf_idx_counter[0], max_seq_len - 1)
            leaf_idx_counter[0] += 1
            return node

        children = [nltk_to_custom(child, leaf_idx_counter) for child in nltk_tree]
        return TreeNode(nltk_tree.label(), children=children, is_leaf=False)

    try:
        # benepar expects a string or a list of words. 
        tree = parser.parse(text)
        return nltk_to_custom(tree, [0])
    except Exception:
        # Fallback for empty or unparseable strings
        node = TreeNode(text, is_leaf=True)
        node.index = 0
        return node

class TreeNode:
    """Represents a node in the parse tree."""
    def __init__(self, text, children=None, is_leaf=False):
        self.text = text
        self.children = children if children else []
        self.is_leaf = is_leaf
        self.hidden = None
        self.index = None  # For leaf nodes, the word index


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
            # Leaf node case: child sums are zero vectors
            batch_size = x.size(0)
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
                feat = leaf_features[node.index].unsqueeze(0)  # (batch, feature_dim)
                h, c = self.cell(feat, [], [])
                node.hidden = h
                node.cell = c
                return h, c
            else:
                # Internal node: compose children
                child_h = []
                child_c = []
                for child in node.children:
                    process_node(child)
                    child_h.append(child.hidden)
                    child_c.append(child.cell)

                child_c_sum = sum(child_c)

                # FIX: Internal nodes in a constituent tree don't have their own word inputs. 
                # We pass a zero-tensor of the expected input_size (feature_dim)
                feat_dim = leaf_features.size(-1)
                x_empty = torch.zeros(1, feat_dim, device=leaf_features.device)

                h, c = self.cell(x_empty, child_h, child_c_sum)
                
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

        # Tree-LSTM
        self.tree_lstm = TreeLSTM(cnn_output_dim, hidden_size)

        # Output layer for VAD regression
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)  # V, A, D
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, trees, leaf_indices):
        """
        Updated forward pass to utilize internal embeddings and CNN.
        Args:
            x: (batch, seq_len) word indices
            trees: list of TreeNode objects (batch)
            leaf_indices: list of lists, word indices for each tree's leaves
        """
        batch_size, seq_len = x.size()

        # 1. Generate text embeddings
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # 2. Extract Regional CNN features
        cnn_features = self.cnn(embedded)  # (batch, cnn_output_dim)
        
        # Expand CNN features to match sequence length (simulating phrase-level broadcast)
        cnn_expanded = cnn_features.unsqueeze(1).expand(batch_size, seq_len, cnn_features.size(1))

        # 3. Hierarchical composition via Tree-LSTM
        root_hiddens = []
        for i, tree in enumerate(trees):
            # Pass the expanded CNN features for this specific batch item
            tree_leaf_features = cnn_expanded[i] 
            root_h = self.tree_lstm(tree, tree_leaf_features)
            root_hiddens.append(root_h)

        root_hidden = torch.cat(root_hiddens, dim=0)  # (batch, hidden_size)
        root_hidden = self.dropout(root_hidden)

        # 4. Predict VAD
        vad_pred = self.fc(root_hidden)  

        return vad_pred


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

        # 1. Standard tokenization for the CNN
        words = self._tokenize(text)
        indices = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]
        
        # Determine actual sequence length before padding
        actual_len = min(len(indices), self.max_seq_len)

        # Pad sequence
        if len(indices) > self.max_seq_len:
            indices = indices[:self.max_seq_len]
        else:
            indices = indices + [self.word2idx['<PAD>']] * (self.max_seq_len - len(indices))

        x = torch.tensor(indices, dtype=torch.long)

        # 2. Generate the parse tree
        tree = text_to_treenode(text, self.max_seq_len)
        
        # 3. Generate leaf indices list (0 to actual_len - 1)
        leaf_indices = list(range(actual_len))

        return x, vad, tree, leaf_indices


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
            x, vad, trees, leaf_indices = batch  # Unpack 4 items
            x = x.to(device)
            vad = vad.to(device)

            optimizer.zero_grad()
            pred = model(x, trees, leaf_indices)  # Pass trees to model
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
                x, vad, trees, leaf_indices = batch
                x = x.to(device)
                vad = vad.to(device)
                pred = model(x, trees, leaf_indices)
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

def load_glove_embeddings(glove_txt_path, word2idx, embed_dim=300):
    # Define the path for the fast-loading numpy file
    saved_matrix_path = 'glove_embeddings_cached.npy'

    # 1. Check if we already parsed and saved it
    if os.path.exists(saved_matrix_path):
        print(f"Loading cached embeddings from {saved_matrix_path}...")
        matrix = np.load(saved_matrix_path)
        return torch.FloatTensor(matrix)

    # 2. If not, we parse the massive text file
    print(f"Parsing raw GloVe file from {glove_txt_path}...")
    matrix = np.random.normal(scale=0.1, size=(len(word2idx), embed_dim))
    matrix[word2idx.get('<PAD>', 0)] = 0.0  # Zero out the padding index

    found_words = 0
    with open(glove_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().split(' ')
            word = " ".join(values[:-300])
            
            if word in word2idx:
                vector = np.asarray(values[-300:], dtype='float32')
                matrix[word2idx[word]] = vector
                found_words += 1

    print(f"Found {found_words} out of {len(word2idx)} words in GloVe.")

    # 3. Save the resulting matrix for next time!
    print(f"Saving parsed matrix to {saved_matrix_path} for faster loading...")
    np.save(saved_matrix_path, matrix)

    return torch.FloatTensor(matrix)

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

    # Load the GloVe matrix
    glove_file_path = 'glove.840B.300d-003.txt'
    pretrained_matrix = load_glove_embeddings(glove_file_path, dataset.word2idx, EMBED_DIM)

    # Load custom fine-tuned matrix directly. NOTE: set freeze_embeddings=True when using this.
    # Comment out the 
    # Load only after training the model with freeze_embeddings=False to allow the embeddings to adapt, then save the adapted matrix and load it here for a second round of training with freeze_embeddings=True.
    # print("Loading custom fine-tuned embeddings...")
    # custom_matrix = np.load('emobank_custom_embeddings2.npy')

    # Create model
    model = TreeStructuredRCNNLSTM(
        vocab_size=dataset.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        num_filters=NUM_FILTERS,
        filter_sizes=FILTER_SIZES,
        dropout=0.5,
        pretrained_embeddings=pretrained_matrix,
        freeze_embeddings=True
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
        lr=LEARNING_RATE,
        
    )
    # Extract the learned weights back into a NumPy array
    adapted_matrix = model.embedding.weight.detach().cpu().numpy()
    
    # Save them to a new file
    np.save('emobank_custom_embeddings2.npy', adapted_matrix)
    print("Saved the custom embeddings!")

    

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
    results_df.to_csv('model2_predictions.csv', index=False)
    print("\nPredictions saved to model2_predictions.csv")


if __name__ == '__main__':
    main()
