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

# Ensure parser is ready
benepar.download('benepar_en3')
parser = benepar.Parser("benepar_en3")

def load_glove_embeddings(glove_txt_path, word2idx, embed_dim=300):
    """Loads GloVe embeddings and caches them as a fast .npy file."""
    saved_matrix_path = 'glove_embeddings_cached.npy'

    if os.path.exists(saved_matrix_path):
        print(f"Loading cached embeddings from {saved_matrix_path}...")
        matrix = np.load(saved_matrix_path)
        return torch.FloatTensor(matrix)

    print(f"Parsing raw GloVe file from {glove_txt_path}...")
    matrix = np.random.normal(scale=0.1, size=(len(word2idx), embed_dim))
    matrix[word2idx.get('<PAD>', 0)] = 0.0 

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
    print(f"Saving parsed matrix to {saved_matrix_path} for faster loading...")
    np.save(saved_matrix_path, matrix)

    return torch.FloatTensor(matrix)


def custom_collate_fn(batch):
    """Custom collate for batching external parse trees."""
    x = torch.stack([item[0] for item in batch])
    vad = torch.stack([item[1] for item in batch])
    trees = [item[2] for item in batch]
    return x, vad, trees


def text_to_treenode(text, max_seq_len):
    """Builds the exact TreeNode hierarchy the authors intended."""
    def nltk_to_custom(nltk_tree, leaf_idx_counter):
        if isinstance(nltk_tree, str):
            node = TreeNode(nltk_tree, is_leaf=True)
            node.index = min(leaf_idx_counter[0], max_seq_len - 1)
            leaf_idx_counter[0] += 1
            return node

        children = [nltk_to_custom(child, leaf_idx_counter) for child in nltk_tree]
        return TreeNode(nltk_tree.label(), children=children, is_leaf=False)

    try:
        tree = parser.parse(text)
        return nltk_to_custom(tree, [0])
    except Exception:
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
        self.index = None  


class RegionalCNN(nn.Module):
    def __init__(self, embed_dim, num_filters=100, filter_sizes=(3, 4, 5)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs)
            for fs in filter_sizes
        ])

    def forward(self, x, mask=None):
        x = x.transpose(1, 2) 
        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  
            pooled = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) 
            pooled_outputs.append(pooled)
        return torch.cat(pooled_outputs, dim=1) 


class TreeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    def forward(self, x, child_h, child_c):
        # FIX: Ensure leaf nodes have zero-tensors to prevent UnboundLocalError
        if len(child_h) == 0:
            batch_size = x.size(0)
            child_h_sum = torch.zeros(batch_size, self.hidden_size, device=x.device)
            child_c_sum = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            child_h_sum = sum(child_h)
            child_c_sum = sum(child_c)

        combined = torch.cat([child_h_sum, x], dim=1)
        gates = self.W(combined)
        gates = gates.chunk(4, dim=1)

        f = torch.sigmoid(gates[0]) 
        i = torch.sigmoid(gates[1])  
        c_tilde = torch.tanh(gates[2])  
        o = torch.sigmoid(gates[3])  

        c = f * child_c_sum + i * c_tilde
        h = o * torch.tanh(c)
        return h, c


class TreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = TreeLSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, tree, leaf_features):
        """Processes the exact tree layout requested in the README."""
        def process_node(node):
            if node.is_leaf:
                # Fetches the specific word tensor for this leaf
                feat = leaf_features[node.index]  
                h, c = self.cell(feat, [], [])
                node.hidden = h
                node.cell = c
                return h, c
            else:
                child_h = []
                child_c = []
                for child in node.children:
                    process_node(child)
                    child_h.append(child.hidden)
                    child_c.append(child.cell)

                child_h_avg = torch.stack(child_h).mean(dim=0)
                child_c_sum = sum(child_c)

                h, c = self.cell(child_h_avg, child_h, child_c_sum)
                node.hidden = h
                node.cell = c
                return h, c

        root_h, _ = process_node(tree)
        return root_h


class TreeStructuredRCNNLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=300,
        hidden_size=150,
        num_filters=100,
        filter_sizes=(3, 4, 5),
        dropout=0.5,
        pretrained_embeddings=None,
        freeze_embeddings=False
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.cnn = RegionalCNN(embed_dim, num_filters, filter_sizes)
        cnn_output_dim = num_filters * len(filter_sizes)

        # Projection layer to merge Word Embedding and CNN context
        self.leaf_proj = nn.Linear(embed_dim + cnn_output_dim, hidden_size)

        self.tree_lstm = TreeLSTM(hidden_size, hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3) 
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, trees):
        batch_size, seq_len = x.size()

        # 1. Base features
        embedded = self.embedding(x)  
        cnn_features = self.cnn(embedded) 
        
        # 2. Expand and concatenate to give every word full context
        cnn_expanded = cnn_features.unsqueeze(1).expand(batch_size, seq_len, cnn_features.size(1))
        combined_features = torch.cat([embedded, cnn_expanded], dim=2) 

        # 3. Execute the Authors' README Approach
        root_hiddens = []
        for i, tree in enumerate(trees):
            # Prepare the leaf_features list exactly as TreeLSTM expects
            # unsqueeze(0) ensures the batch dimension is 1 for the individual tree nodes
            leaf_features_for_tree = [self.leaf_proj(combined_features[i, j]).unsqueeze(0) for j in range(seq_len)]
            
            # The exact call demonstrated in the README file
            root_h = self.tree_lstm(tree, leaf_features_for_tree)
            root_hiddens.append(root_h.squeeze(0))

        root_hidden = torch.stack(root_hiddens, dim=0) 
        root_hidden = self.dropout(root_hidden)

        return self.fc(root_hidden)


class EmoBankDataset(Dataset):
    def __init__(self, csv_path, vocab=None, word2idx=None, max_seq_len=100):
        self.data = pd.read_csv(csv_path)
        self.max_seq_len = max_seq_len

        if word2idx is None:
            self.word2idx = {'<PAD>': 0, '<UNK>': 1}
            self._build_vocab()
        else:
            self.word2idx = word2idx
        self.vocab_size = len(self.word2idx)

    def _build_vocab(self):
        word_counts = defaultdict(int)
        for text in self.data['text']:
            words = self._tokenize(text)
            for word in words:
                word_counts[word] += 1
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        for word, _ in sorted_words[:30000]:  
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)

    def _tokenize(self, text):
        if pd.isna(text): return []
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        vad = torch.tensor([row['V'], row['A'], row['D']], dtype=torch.float32)

        words = self._tokenize(text)
        indices = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]
        
        if len(indices) > self.max_seq_len:
            indices = indices[:self.max_seq_len]
        else:
            indices = indices + [self.word2idx['<PAD>']] * (self.max_seq_len - len(indices))

        x = torch.tensor(indices, dtype=torch.long)
        tree = text_to_treenode(text, self.max_seq_len)
        
        return x, vad, tree


def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x, vad, trees = batch 
            x, vad = x.to(device), vad.to(device)

            optimizer.zero_grad()
            pred = model(x, trees) 
            loss = criterion(pred, vad)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, vad, trees = batch
                x, vad = x.to(device), vad.to(device)
                pred = model(x, trees)
                loss = criterion(pred, vad)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load('best_model.pt'))
    return model


def main():
    DATA_PATH = './emobank.csv'
    GLOVE_PATH = './glove.840B.300d-003.txt'
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

    dataset = EmoBankDataset(DATA_PATH, max_seq_len=MAX_SEQ_LEN)
    print(f"Vocabulary size: {dataset.vocab_size}")

    train_indices, val_indices, test_indices = [], [], []
    for idx, row in enumerate(dataset.data.itertuples()):
        if row.split == 'train': train_indices.append(idx)
        elif row.split == 'dev': val_indices.append(idx)
        elif row.split == 'test': test_indices.append(idx)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

    # Automatically loads/caches the GloVe vectors
    pretrained_matrix = load_glove_embeddings(GLOVE_PATH, dataset.word2idx, EMBED_DIM)

    model = TreeStructuredRCNNLSTM(
        vocab_size=dataset.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        num_filters=NUM_FILTERS,
        filter_sizes=FILTER_SIZES,
        dropout=0.5,
        pretrained_embeddings=pretrained_matrix,
        freeze_embeddings=False # Set to false so it learns the dataset domain
    ).to(device)

    model = train_model(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LEARNING_RATE)

    # Save the fine-tuned custom embeddings for future use
    np.save('emobank_custom_embeddings.npy', model.embedding.weight.detach().cpu().numpy())
    print("Saved the fine-tuned custom embeddings to emobank_custom_embeddings.npy")

    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, vad, trees = batch
            x = x.to(device)
            pred = model(x, trees)
            predictions.append(pred.cpu().numpy())
            targets.append(vad.numpy())

    predictions = np.vstack(predictions)
    targets = np.vstack(targets)

    mse = np.mean((predictions - targets) ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets), axis=0)
    correlation = np.array([np.corrcoef(predictions[:, i], targets[:, i])[0, 1] for i in range(3)])

    print("\n=== Test Results ===")
    print(f"Valence  - MSE: {mse[0]:.4f}, RMSE: {rmse[0]:.4f}, MAE: {mae[0]:.4f}, Corr: {correlation[0]:.4f}")
    print(f"Arousal  - MSE: {mse[1]:.4f}, RMSE: {rmse[1]:.4f}, MAE: {mae[1]:.4f}, Corr: {correlation[1]:.4f}")
    print(f"Dominance- MSE: {mse[2]:.4f}, RMSE: {rmse[2]:.4f}, MAE: {mae[2]:.4f}, Corr: {correlation[2]:.4f}")

    results_df = dataset.data.iloc[test_indices].copy()
    results_df[['V_pred', 'A_pred', 'D_pred']] = predictions
    results_df.to_csv('predictions3.csv', index=False)
    print("\nPredictions saved to predictions3.csv")

if __name__ == '__main__':
    main()