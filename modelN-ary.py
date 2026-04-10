"""
Tree-Structured Regional CNN-LSTM for VAD Prediction
Architecture: True Regional CNN paired with N-ary (Binary) Constituency Parse Trees.
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
import pickle
import benepar
import nltk

# Ensure parser and NLTK tree functionality are ready
benepar.download('benepar_en3')
parser = benepar.Parser("benepar_en3")

def load_glove_embeddings(glove_txt_path, word2idx, embed_dim=300):
    saved_matrix_path = 'glove_embeddings_cached.npy'
    if os.path.exists(saved_matrix_path):
        print(f"Loading cached embeddings from {saved_matrix_path}...")
        return torch.FloatTensor(np.load(saved_matrix_path))

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
    np.save(saved_matrix_path, matrix)
    return torch.FloatTensor(matrix)


def custom_collate_fn(batch):
    x = torch.stack([item[0] for item in batch])
    vad = torch.stack([item[1] for item in batch])
    trees = [item[2] for item in batch]
    return x, vad, trees


class TreeNode:
    def __init__(self, text, children=None, is_leaf=False):
        self.text = text
        self.children = children if children else []
        self.is_leaf = is_leaf
        self.hidden = None
        self.index = None  

def text_to_treenode(text, max_seq_len):
    """Builds a Binary Constituency Tree using NLTK's Chomsky Normal Form."""
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
        tree.chomsky_normal_form(factor='right')
        return nltk_to_custom(tree, [0])
    except Exception:
        node = TreeNode(text, is_leaf=True)
        node.index = 0
        return node


def precompute_trees(csv_path, max_seq_len, save_path='emobank_constituency_trees_cached.pkl'):
    if os.path.exists(save_path):
        print(f"Loading pre-parsed Binary Constituency trees from {save_path}...")
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    print(f"No cached trees found. Parsing dataset with benepar ({csv_path})...")
    data = pd.read_csv(csv_path)
    trees = []
    
    total = len(data)
    for i, text in enumerate(data['text']):
        if i > 0 and i % 1000 == 0:
            print(f"Parsed {i}/{total} sentences...")
        trees.append(text_to_treenode(text, max_seq_len))
        
    print("Parsing complete! Saving trees to disk...")
    with open(save_path, 'wb') as f:
        pickle.dump(trees, f)
    return trees


# --- MODIFIED: True Regional CNN ---
class RegionalCNN(nn.Module):
    def __init__(self, embed_dim, num_filters=100, filter_sizes=(3, 4, 5)):
        super().__init__()
        # padding='same' ensures the CNN output length perfectly matches the sentence length.
        # It creates a localized feature vector specifically centered around each individual word.
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs, padding='same')
            for fs in filter_sizes
        ])

    def forward(self, x, mask=None):
        # x is (batch, seq_len, embed_dim)
        x = x.transpose(1, 2) # (batch, embed_dim, seq_len)
        
        regional_outputs = []
        for conv in self.convs:
            # We NO LONGER apply max_pool1d. The convolution itself is our regional feature.
            # Output is strictly (batch, num_filters, seq_len)
            conv_out = F.relu(conv(x))  
            regional_outputs.append(conv_out)
            
        # Concatenate the different filter sizes
        combined_regional = torch.cat(regional_outputs, dim=1) 
        
        # Transpose back so it matches the standard (batch, seq_len, total_filters) format
        return combined_regional.transpose(1, 2) 


class BinaryTreeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_x = nn.Linear(input_size, 5 * hidden_size)
        self.U_L = nn.Linear(hidden_size, 5 * hidden_size, bias=False)
        self.U_R = nn.Linear(hidden_size, 5 * hidden_size, bias=False)

    def forward(self, x, left_h, left_c, right_h, right_c):
        gates = self.W_x(x)
        if left_h is not None:
            gates += self.U_L(left_h)
        if right_h is not None:
            gates += self.U_R(right_h)
            
        i, f_L, f_R, o, u = gates.chunk(5, dim=1)
        
        i, f_L, f_R, o = torch.sigmoid(i), torch.sigmoid(f_L), torch.sigmoid(f_R), torch.sigmoid(o)
        u = torch.tanh(u)
        
        c = i * u
        if left_c is not None:
            c += f_L * left_c
        if right_c is not None:
            c += f_R * right_c
            
        h = o * torch.tanh(c)
        return h, c


class TreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = BinaryTreeLSTMCell(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, tree, leaf_features):
        device = leaf_features[0].device
        
        def process_node(node):
            if getattr(node, 'hidden', None) is not None:
                return node.hidden, node.cell
                
            if node.is_leaf:
                feat = leaf_features[node.index]
                h, c = self.cell(feat, None, None, None, None)
            else:
                feat = torch.zeros(1, self.input_size, device=device)
                
                left_h, left_c = None, None
                right_h, right_c = None, None
                
                if len(node.children) > 0:
                    left_h, left_c = process_node(node.children[0])
                if len(node.children) > 1:
                    right_h, right_c = process_node(node.children[1])
                    
                h, c = self.cell(feat, left_h, left_c, right_h, right_c)
                
            node.hidden = h
            node.cell = c
            return h, c

        root_h, _ = process_node(tree)
        return root_h


class TreeStructuredRCNNLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_size=150, num_filters=100, filter_sizes=(3, 4, 5), dropout=0.5, pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.cnn = RegionalCNN(embed_dim, num_filters, filter_sizes)
        cnn_output_dim = num_filters * len(filter_sizes)

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

        # 1. The Word is embedded
        embedded = self.embedding(x)  
        
        # 2. The CNN extracts the local neighborhood feature for every individual word
        cnn_features = self.cnn(embedded) 
        
        # --- MODIFIED: Direct word-to-word concatenation ---
        # No sequence expansion needed. Word[i] is cleanly paired with CNN_Feature[i]
        combined_features = torch.cat([embedded, cnn_features], dim=2) 

        root_hiddens = []
        for i, tree in enumerate(trees):
            leaf_features_for_tree = [self.leaf_proj(combined_features[i, j]).unsqueeze(0) for j in range(seq_len)]
            root_h = self.tree_lstm(tree, leaf_features_for_tree)
            root_hiddens.append(root_h.squeeze(0))

        root_hidden = torch.stack(root_hiddens, dim=0) 
        root_hidden = self.dropout(root_hidden)

        return self.fc(root_hidden)


class EmoBankDataset(Dataset):
    def __init__(self, csv_path, precomputed_trees, vocab=None, word2idx=None, max_seq_len=100):
        self.data = pd.read_csv(csv_path)
        self.max_seq_len = max_seq_len
        self.trees = precomputed_trees  

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
        tree = self.trees[idx] 
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

    cached_trees = precompute_trees(DATA_PATH, MAX_SEQ_LEN, save_path='emobank_constituency_trees_cached.pkl')
    
    dataset = EmoBankDataset(DATA_PATH, precomputed_trees=cached_trees, max_seq_len=MAX_SEQ_LEN)
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

    pretrained_matrix = load_glove_embeddings(GLOVE_PATH, dataset.word2idx, EMBED_DIM)

    model = TreeStructuredRCNNLSTM(
        vocab_size=dataset.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        num_filters=NUM_FILTERS,
        filter_sizes=FILTER_SIZES,
        dropout=0.5,
        pretrained_embeddings=pretrained_matrix,
        freeze_embeddings=False
    ).to(device)

    model = train_model(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LEARNING_RATE)

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

if __name__ == '__main__':
    main()