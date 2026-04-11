# To run this script, first create a a virtual environment for python by typing in your terminal:
# python -m venv myenv
# Then activate it (powershell): .\myenv\Scripts\Activate.ps1
# Then install the required packages: pip install -r requirements2.txt
# Make sure you have the GloVe file (glove.840B.300d.txt) and the dataset (writer_10240.csv) in the same directory as this script, or provide the correct paths via command line arguments.



import os
import argparse
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import stanza
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from tqdm import tqdm
import pickle
import hashlib
import json


def build_full_vocab_embeddings(glove_path, embed_dim=300):
    print("Loading full GloVe vocabulary (2.2M words)...")
    wv = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    embeddings = [np.zeros(embed_dim), np.random.uniform(-0.05, 0.05, embed_dim)]
    for word in wv.key_to_index:
        vocab[word] = len(vocab)
        embeddings.append(wv[word])
    embedding_matrix = np.stack(embeddings).astype(np.float32)
    return vocab, embedding_matrix

def get_embedding_cache_path(config, all_regions):
    """
    Create a unique cache path for vocabulary and embeddings based on:
    - dataset name
    - data file modification time (or hash)
    - region extraction depth
    - glove path
    - max_regions_perc, max_region_len_perc (indirectly via regions)
    """
    cache_dir = os.path.dirname(config.get('cache_file', '/content/drive/MyDrive/emobank/cache/glove_embedding_cache.pkl')) or 'cache'
    os.makedirs(cache_dir, exist_ok=True)

    # Use modification time of data file as part of key
    data_path = config['data_path']
    if os.path.exists(data_path):
        mtime = os.path.getmtime(data_path)
    else:
        mtime = 0

    # Create a hash of key parameters
    key_dict = {
        'dataset': config['dataset'],
        'data_path': data_path,
        'mtime': mtime,
        'depth': config.get('depth', 3),
        'glove_path': config.get('glove_path', ''),
        'embed_dim': config.get('embed_dim', 300),
        'num_regions': len(all_regions),  # number of samples
        'region_counts': [len(r) for r in all_regions[:10]],  # first few to detect changes
    }
    key_str = json.dumps(key_dict, sort_keys=True)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()[:12]

    return os.path.join(cache_dir, f'embed_cache_{key_hash}.pkl')

# -------------------------------
# GPU Configuration
# -------------------------------
def configure_gpu(force_cpu=False):
    """Detect GPUs, set memory growth, and print device info."""
    if force_cpu:
        tf.config.set_visible_devices([], 'GPU')
        print("Forced CPU mode. Visible GPUs: disabled")
        return

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Memory growth configuration error: {e}")
        print("GPU memory growth enabled.")
    else:
        print("No GPU found. Running on CPU.")

# -------------------------------
# Stanza Parser (with constituency)
# -------------------------------
stanza.download('en')
# Pipeline will be initialized in run_training

# -------------------------------
# 2. Tree Binarization (right-branching)
# -------------------------------
def binarize_tree(tree):
    if len(tree.children) == 0:
        return tree
    elif len(tree.children) == 1:
        child = binarize_tree(tree.children[0])
        return type(tree)(label=tree.label, children=[child])
    elif len(tree.children) == 2:
        left = binarize_tree(tree.children[0])
        right = binarize_tree(tree.children[1])
        return type(tree)(label=tree.label, children=[left, right])
    else:
        first = tree.children[0]
        rest = tree.children[1:]
        right_child = type(tree)(label=tree.label, children=rest)
        right_binarized = binarize_tree(right_child)
        left_binarized = binarize_tree(first)
        return type(tree)(label=tree.label, children=[left_binarized, right_binarized])

# -------------------------------
# 3. Region Extraction by Depth
# -------------------------------
def extract_regions(tree, target_depth, current_depth=1):
    regions = []
    if current_depth == target_depth:
        regions.append(tree.leaf_labels())
    else:
        for child in tree.children:
            regions.extend(extract_regions(child, target_depth, current_depth + 1))
    return regions

# -------------------------------
# 4. Parse a Single Text
# -------------------------------
def parse_and_extract_regions(text, depth, nlp_pipeline):
    doc = nlp_pipeline(text)
    all_regions = []
    for sent in doc.sentences:
        if len(sent.constituency.children) == 0:
            continue
        bin_tree = binarize_tree(sent.constituency)
        regions = extract_regions(bin_tree, target_depth=depth)
        all_regions.extend(regions)
    return all_regions

# -------------------------------
# 5. Build Vocabulary and GloVe Embeddings
# -------------------------------
def build_vocab_and_embeddings(all_region_word_lists, glove_path, embed_dim=300):
    print("Loading GloVe...")
    wv = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)

    vocab = {'<PAD>': 0, '<UNK>': 1}
    embeddings = [np.zeros(embed_dim), np.random.uniform(-0.05, 0.05, embed_dim)]

    for regions in all_region_word_lists:
        for region in regions:
            for word in region:
                if word not in vocab:
                    vocab[word] = len(vocab)
                    if word in wv:
                        embeddings.append(wv[word])
                    else:
                        embeddings.append(np.random.uniform(-0.05, 0.05, embed_dim))

    embedding_matrix = np.stack(embeddings).astype(np.float32)
    return vocab, embedding_matrix

# -------------------------------
# 6. Build Tensor from Region Word Lists
# -------------------------------
def build_tensor(all_regions, vocab, max_regions, max_region_len):
    n_samples = len(all_regions)
    X = np.zeros((n_samples, max_regions, max_region_len), dtype=np.int32)
    mask = np.zeros((n_samples, max_regions), dtype=np.bool_)

    for i, regions in enumerate(all_regions):
        num_regions = min(len(regions), max_regions)
        mask[i, :num_regions] = True
        for j, region in enumerate(regions[:max_regions]):
            for k, word in enumerate(region[:max_region_len]):
                X[i, j, k] = vocab.get(word, vocab['<UNK>'])
    return X, mask

# -------------------------------
# 7. Model Architecture
# -------------------------------
def build_regional_cnn_lstm(vocab_size, embed_dim, embedding_matrix,
                            max_region_len, max_regions, num_outputs,
                            cnn_filters=60, filter_length=3, pool_length=2,
                            lstm_units=120, recurrent_dropout=0.25, spatial_dropout=0.2):
    region_input = layers.Input(shape=(max_regions, max_region_len), name='region_input')
    mask_input = layers.Input(shape=(max_regions,), dtype='bool', name='mask_input')

    embedding = layers.Embedding(vocab_size, embed_dim,
                                 weights=[embedding_matrix],
                                 trainable=True, mask_zero=False,
                                 name='word_embedding')
    embedded = layers.TimeDistributed(embedding)(region_input)
    embedded = layers.TimeDistributed(layers.SpatialDropout1D(spatial_dropout))(embedded)

    def region_cnn():
        model = keras.Sequential([
            layers.Conv1D(filters=cnn_filters, kernel_size=filter_length,
                          activation='relu', padding='valid'),   # ← 'valid'
            layers.MaxPooling1D(pool_size=pool_length, padding='valid'),
            layers.Flatten()
        ])
        return model

    region_vectors = layers.TimeDistributed(region_cnn())(embedded)

    mask_float = layers.Lambda(lambda x: tf.cast(x, tf.float32))(mask_input)
    mask_expanded = layers.Reshape((max_regions, 1))(mask_float)
    masked_region_vectors = layers.Multiply()([region_vectors, mask_expanded])

    lstm_out = layers.LSTM(lstm_units, dropout=0.0, recurrent_dropout=recurrent_dropout,
                           return_sequences=False, name='lstm')(masked_region_vectors)

    outputs = [layers.Dense(1, activation='linear', name=f'target_{i}')(lstm_out)
               for i in range(num_outputs)]

    model = Model(inputs=[region_input, mask_input], outputs=outputs)
    return model


# ============================================================
# Dataset Abstraction
# ============================================================
class BaseDataset(ABC):
    def __init__(self, data_path):
        self.data_path = data_path
        self._texts = None
        self._targets = None
        self._target_names = None

    @abstractmethod
    def load_raw_data(self):
        pass

    @abstractmethod
    def extract_texts(self, df):
        pass

    @abstractmethod
    def extract_targets(self, df):
        pass

    def prepare(self):
      df = self.load_raw_data()
      self._texts = self.extract_texts(df)
      self._targets, self._target_names = self.extract_targets(df)
      # If there was a validity mask, filter texts
      if hasattr(self, '_valid_mask'):
          mask = self._valid_mask
          self._texts = [t for t, m in zip(self._texts, mask) if m]
          # Reset index to avoid misalignment

    def get_texts(self):
        return self._texts

    def get_targets(self):
        return self._targets

    def get_target_names(self):
        return self._target_names


class EmobankDataset(BaseDataset):
    """Dataset for EmoBank writer_10240.csv"""
    def load_raw_data(self):
        return pd.read_csv(self.data_path)

    def extract_texts(self, df):
        return df['text'].astype(str).tolist()

    def extract_targets(self, df):
        targets = df[['V', 'A']].values.astype(np.float32)
        names = ['Valence', 'Arousal']
        return targets, names

class EmobankWriterValenceDataset(BaseDataset):
    def load_raw_data(self):
        return pd.read_csv(self.data_path)

    def extract_texts(self, df):
        return df['text'].astype(str).tolist()

    def extract_targets(self, df):
        # Force numeric conversion, coerce errors to NaN
        v_series = pd.to_numeric(df['V'], errors='coerce')
        # Drop rows with NaN targets
        valid_mask = v_series.notna()
        if not valid_mask.all():
            print(f"Warning: Dropping {len(df) - valid_mask.sum()} rows with invalid V values.")
        # Keep only valid rows (both texts and targets will be filtered later)
        self._valid_mask = valid_mask
        targets = v_series[valid_mask].values.astype(np.float32).reshape(-1, 1)
        names = ['Valence']
        return targets, names


DATASET_REGISTRY = {
    'emobank': EmobankDataset,
    'emobank_valence_writer': EmobankWriterValenceDataset
}


# ============================================================
# Main Training Function (Notebook & CLI friendly)
# ============================================================
def run_training(config):
    """
    config: dict with keys:
        dataset, data_path, glove_path, depth, embed_dim, batch_size,
        epochs, patience, max_regions_perc, max_region_len_perc,
        cache_file, no_gpu
    """
    # ---------- GPU Configuration ----------
    configure_gpu(force_cpu=config.get('no_gpu', False))

    use_gpu_stanza = not config.get('no_gpu', False) and bool(tf.config.list_physical_devices('GPU'))
    print(f"Stanza GPU mode: {use_gpu_stanza}")
    nlp = stanza.Pipeline('en', processors='tokenize,pos,constituency', use_gpu=use_gpu_stanza)

    # ---------- Instantiate dataset ----------
    dataset_class = DATASET_REGISTRY[config['dataset']]
    dataset = dataset_class(config['data_path'])
    dataset.prepare()
    texts = dataset.get_texts()
    y = dataset.get_targets()
    if y.dtype == np.object_:
        print("Converting targets to float32...")
        y = y.astype(np.float32)
    target_names = dataset.get_target_names()
    num_outputs = y.shape[1]
    print(f"Loaded {len(texts)} samples with {num_outputs} targets: {target_names}")

    # ---------- Parse Texts (or load from cache) ----------
    cache_file = config.get('cache_file', 'parsed_emobank_writer.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            all_regions = pickle.load(f)
        print("Loaded parsed regions from cache.")
    else:
        all_regions = []
        for text in tqdm(texts, desc="Parsing texts"):
            regions = parse_and_extract_regions(text, depth=config['depth'], nlp_pipeline=nlp)
            all_regions.append(regions)
        cache_dir = os.path.dirname(cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(all_regions, f)
        print("Parsed and cached regions.")

        # ---------- Vocabulary & Embeddings (with caching) ----------
    embed_cache_path = get_embedding_cache_path(config, all_regions)
    embed_matrix_cache = embed_cache_path.replace('.pkl', '_matrix.npy')

    if os.path.exists(embed_cache_path) and os.path.exists(embed_matrix_cache):
        print("Loading full GloVe vocabulary and embeddings from cache...")
        with open(embed_cache_path, 'rb') as f:
            vocab = pickle.load(f)
        embed_matrix = np.load(embed_matrix_cache)
        vocab_size = len(vocab)
        print(f"Vocabulary size: {vocab_size}")
    else:
        print("Building full GloVe vocabulary (2.2M words, this may take a while)...")
        vocab, embed_matrix = build_full_vocab_embeddings(config['glove_path'], config.get('embed_dim', 300))
        vocab_size = len(vocab)
        print(f"Vocabulary size: {vocab_size}")
        # Save to cache
        with open(embed_cache_path, 'wb') as f:
            pickle.dump(vocab, f)
        np.save(embed_matrix_cache, embed_matrix)
        print(f"Embedding cache saved to {embed_cache_path}")

    # ---------- Determine Padding Sizes ----------
    region_counts = [len(r) for r in all_regions]
    max_regions = int(np.percentile(region_counts, config.get('max_regions_perc', 98)))
    token_counts = [len(w) for regions in all_regions for region in regions for w in region]
    max_region_len = int(np.percentile(token_counts, config.get('max_region_len_perc', 98)))
    print(f"Max regions: {max_regions}, Max region length: {max_region_len}")

    # ---------- Build Tensor ----------
    X, mask = build_tensor(all_regions, vocab, max_regions, max_region_len)

    # ---------- Train/Val/Test Split ----------
    X_train, X_temp, mask_train, mask_temp, y_train, y_temp = train_test_split(
        X, mask, y, test_size=0.2, random_state=42)
    X_val, X_test, mask_val, mask_test, y_val, y_test = train_test_split(
        X_temp, mask_temp, y_temp, test_size=0.5, random_state=42)

    # ---------- Build Model ----------
    model = build_regional_cnn_lstm(vocab_size, config.get('embed_dim', 300), embed_matrix,
                                   max_region_len, max_regions, num_outputs)

    # ---------- Compile ----------
    if num_outputs == 1:
        model.compile(optimizer=keras.optimizers.Adam(), loss='mse')
        train_targets = y_train
        val_targets = y_val
    else:
        losses = {f'target_{i}': 'mse' for i in range(num_outputs)}
        loss_weights = {f'target_{i}': 1.0/num_outputs for i in range(num_outputs)}
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=losses, loss_weights=loss_weights)
        train_targets = {f'target_{i}': y_train[:, i] for i in range(num_outputs)}
        val_targets = {f'target_{i}': y_val[:, i] for i in range(num_outputs)}

    model.summary()

    # ---------- Callbacks ----------
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.get('patience', 5),
                                               restore_best_weights=True)

    # ---------- Train ----------
    history = model.fit(
        [X_train, mask_train], train_targets,
        validation_data=([X_val, mask_val], val_targets),
        batch_size=config.get('batch_size', 32), epochs=config.get('epochs', 20),
        callbacks=[early_stop], verbose=1)

    # ---------- Evaluate ----------
    preds = model.predict([X_test, mask_test])
    if num_outputs == 1:
        preds = [preds]

    for i, name in enumerate(target_names):
        true = y_test[:, i]
        pred = preds[i].flatten()
        try:
            mae = mean_absolute_error(true, pred)
            pearson_corr, _ = pearsonr(true, pred)
        except Exception as e:
            print(f"Evaluation failed for {name}: {e}")
            mae = np.nan
            pearson_corr = np.nan
        print(f"{name} - MAE: {mae:.4f}, Pearson r: {pearson_corr:.4f}")

    return model, history


# ============================================================
# Command-Line Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train Regional CNN-LSTM on a dataset")
    parser.add_argument('--dataset', type=str, required=True, default='emobank',
                        choices=DATASET_REGISTRY.keys())
    parser.add_argument('--data_path', type=str, required=True, default='./writer_10240.csv',)
    parser.add_argument('--glove_path', type=str,  default='./glove.840B.300d.txt', required=True)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_regions_perc', type=int, default=98)
    parser.add_argument('--max_region_len_perc', type=int, default=98)
    parser.add_argument('--cache_file', type=str, default='./cache/parsed_emobank_writer.pkl',)
    parser.add_argument('--no_gpu', action='store_true')
    args = parser.parse_args()
    config = vars(args)
    run_training(config)


if __name__ == "__main__":
    # If in IPython/Jupyter, provide a convenience example
    try:
        get_ipython()
        print("Notebook environment detected. Use run_training(config_dict) to train.")
        print("Example config:")
        config = {
          'dataset': 'emobank_valence_writer',
          'data_path': './writer_10240.csv',
          'glove_path': './glove.840B.300d.txt',
          'depth': 3,
          'cache_file': './cache/parsed_emobank_writer.pkl',
          'epochs': 20,
          'batch_size': 32
          }
        model, history = run_training(config)
    except NameError:
        # Not in IPython – run command line
        main()