#!/usr/bin/env python3
"""
Tree-Structured Regional CNN-LSTM Model for Dimensional Sentiment Analysis.

Replication of:
    Wang, Yu, Lai, Zhang. "Tree-Structured Regional CNN-LSTM Model for
    Dimensional Sentiment Analysis." IEEE/ACM TASLP, vol. 28, 2020.

Single-file pipeline covering:
    1. Constituency parsing with Stanza (cached).
    2. Tree-depth-based region division.
    3. GloVe 840B / 300d loading (cached per-dataset vocabulary).
    4. Regional CNN-LSTM model built in TensorFlow / Keras.
    5. Training with early stopping + Pearson r / MAE evaluation.

Dataset processors are modular so new datasets can be plugged in by
subclassing ``BaseDatasetProcessor``. The current implementation ships
with ``EmoBankWriterProcessor`` for ``writer_10240.csv``.

5-fold cross-validation is intentionally omitted; the script uses the
paper's 70/20/10 train/dev/test split (Section IV-A).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("regional-cnn-lstm")


# ===========================================================================
# 1. Configuration
# ===========================================================================
BASE_DIR = Path("/home/azureuser/cloudfiles/code/Users/i.eltagonde.533553")


# ===========================================================================
# Dataset-specific optimal region depths
# ===========================================================================
# Per Wang et al. (2020) Section IV-D, the optimal tree-division depth is
# selected on the dev set and differs by dataset and emotion dimension.
# Values recorded here reflect the paper's reported optima; for any new
# dataset added to the pipeline, a dev-set sweep should determine its own
# per-dimension optima and the result recorded in this registry.
DATASET_OPTIMAL_DEPTH: Dict[str, Dict[str, int]] = {
    "emobank_writer": {"valence": 4, "arousal": 5},
    "emobank_reader": {"valence": 4, "arousal": 4},
    "fb": {"valence": 5, "arousal": 5},
    "cvat": {"valence": 8, "arousal": 8},
    "sst": {"valence": 5},  # SST is valence-only; no arousal labels
}


@dataclass
class Config:
    """Single source of truth for paths and hyper-parameters."""

    # --- paths ---------------------------------------------------------
    base_dir: Path = BASE_DIR
    cache_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    glove_path: Path = field(init=False)

    # Optional global override for the embedding file, set from
    # --embedding-path. When non-None, overrides any processor default.
    # Useful for swapping in FastText for Chinese-Word-Vectors, or any
    # other same-format 300-dim file, without editing code.
    embedding_path_override: Optional[Path] = None

    # Optional run identifier. When set, all output artifacts
    # (checkpoints, per-fold metrics, summary JSONs, prediction .npz
    # files) are scoped so they do not overwrite a previous run with a
    # different configuration. Typical use: --run-tag tuned for the
    # default-config run, --run-tag paper for the --paper-config run.
    # When None, filenames/paths match legacy behavior exactly.
    run_tag: Optional[str] = None

    # --- model ---------------------------------------------------------
    embedding_dim: int = 300
    cnn_filters: int = 60
    cnn_kernel_size: int = 3
    pool_size: int = 2
    lstm_units: int = 120

    # --- multi-kernel CNN (optional, off by default) ------------------
    # When enabled, replaces the single-kernel CNN with parallel
    # Conv1D branches at multiple kernel widths (Kim 2014 style).
    # The authors' own published code uses (3, 4, 5) × 100 filters
    # even though the paper text specifies a single kernel of width 3.
    # This is OFF by default to stay faithful to the paper. Turn it
    # on with --multi-kernel.
    multi_kernel: bool = False
    multi_kernel_sizes: Tuple[int, ...] = (3, 4, 5)
    multi_kernel_filters_per_size: int = 100

    # --- regularization ------------------------------------------------
    # Round 4 of tuning:
    #   Round 1 (paper values, 0.25 everywhere):     severe overfit
    #   Round 2 (0.5 everywhere + L2 1e-4):          underfit, σ collapsed
    #   Round 3 (0.3 everywhere, no L2):             overfit after epoch 2
    #   Round 4 (this):                              0.3–0.4, L2 1e-5, LR 5e-4
    embedding_dropout: float = 0.2       # input SpatialDropout1D on GloVe
    spatial_dropout: float = 0.3         # CNN output
    recurrent_dropout: float = 0.3       # LSTM recurrent
    post_lstm_dropout: float = 0.4       # before linear decoder
    l2_reg: float = 1e-5                 # small L2 on conv + dense
    grad_clip_norm: float = 1.0          # Adam clipnorm (0 to disable)

    # --- architecture toggles -----------------------------------------
    bidirectional_lstm: bool = False     # wrap the sequential LSTM with Bidirectional
    trainable_embeddings: bool = False   # frozen GloVe by default

    # --- loss ----------------------------------------------------------
    # "mse" — paper default.
    # "ccc" — 1 − Concordance Correlation Coefficient; standard for VA
    #         regression where MSE collapses predictions to the mean.
    loss_type: str = "mse"

    # --- label handling ------------------------------------------------
    # Standardize V/A to zero-mean, unit-variance per training fold.
    # EmoBank scores cluster tightly around 3.0, which makes MSE loss
    # collapse predictions to the mean (great MAE, terrible Pearson r).
    # Normalizing moves the mean-offset into the decoder bias so the
    # rest of the network is forced to explain variance. Does NOT
    # affect Pearson r (scale-invariant); does affect MSE-loss surface.
    normalize_labels: bool = True

    # --- training ------------------------------------------------------
    batch_size: int = 32
    epochs: int = 40
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-5
    learning_rate: float = 5e-4          # was 1e-3; model was converging too fast

    # --- region division ----------------------------------------------
    region_depth: int = 4                # paper's optimum for EmoBank-writer

    # --- tensor shaping -----------------------------------------------
    # These bound R (max regions per text) and N (max words per region).
    # They are auto-adjusted to the given percentile of the training
    # data when ``auto_shape`` is True. 99.5 covers the tail better
    # than 99 without letting pathological outliers dominate shape.
    max_region_len: int = 20             # N
    max_num_regions: int = 30            # R
    auto_shape: bool = True
    auto_shape_percentile: float = 99.5

    # --- vocabulary ----------------------------------------------------
    # GloVe 840B is CASED — "Happy" and "happy" have distinct vectors
    # (and the cased forms often carry more affective context because
    # they appear at sentence starts). Lower-casing collapses them.
    case_sensitive: bool = False

    # --- target dimension ---------------------------------------------
    # None (default): joint V+A training (Dense(2) decoder, legacy mode)
    # "valence" or "arousal": single-dim training (Dense(1) decoder).
    # Single-dim mode is required for the LSTM vs BiLSTM comparison
    # because optimal region depth differs by dimension (V=4, A=5 on
    # emobank_writer per Wang et al. Section IV-D).
    target_dimension: Optional[str] = None

    # --- splits --------------------------------------------------------
    random_seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1

    # --- special tokens -----------------------------------------------
    pad_token_id: int = 0
    unk_token_id: int = 1

    def __post_init__(self) -> None:
        self.cache_dir = self.base_dir / "cache"
        self.model_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        self.glove_path = self.base_dir / "glove.840B.300d.txt"


# ===========================================================================
# 2. Dataset processors (modular)
# ===========================================================================
class BaseDatasetProcessor:
    """Abstract base for per-dataset preprocessing.

    Subclasses must implement ``load`` (returns a DataFrame),
    ``get_texts`` and ``get_labels``. Keeping this thin keeps the
    pipeline decoupled from dataset-specific column schemas.

    Subclasses may also override the path-construction methods
    (``fold_checkpoint_path``, ``best_model_path``) to control how
    saved artifacts are named for that dataset. The defaults follow
    a stable convention scoped by dataset name, target dimension, and
    model architecture tag, which is sufficient for most datasets.
    """

    # Language code. "en" drives Stanza/embedding defaults; Chinese
    # subclasses override to "zh". Used mainly as a human-readable tag;
    # the actual pipeline behaviour is controlled by
    # stanza_pipeline_kwargs() and embedding_path().
    language: str = "en"

    # Default embedding file within cfg.base_dir. English datasets use
    # Stanford GloVe 840B/300d. Chinese datasets override.
    default_embedding_filename: str = "glove.840B.300d.txt"

    def __init__(self, filepath: Path, name: str) -> None:
        self.filepath = Path(filepath)
        self.name = name

    # hooks -------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_texts(self, df: pd.DataFrame) -> List[str]:
        raise NotImplementedError

    def get_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Return an (N, 2) float32 array ordered as [Valence, Arousal]."""
        raise NotImplementedError

    # parser + embedding hooks -----------------------------------------
    def embedding_path(self, cfg: "Config") -> Path:
        """Return the path to the pre-trained word embedding file for
        this dataset. A global override on Config (set via
        --embedding-path) wins over the processor default, so the same
        class can be reused with a different embedding release without
        editing code.
        """
        if cfg.embedding_path_override is not None:
            return Path(cfg.embedding_path_override)
        return cfg.base_dir / self.default_embedding_filename

    def prepare_for_parser(self, text: str) -> str:
        """Transform a raw dataset text into the string actually fed
        into the Stanza pipeline. Default is identity (English takes
        the raw text directly; Stanza tokenises and sentence-splits).
        Chinese subclasses override to apply Traditional→Simplified
        conversion; Stanza then handles tokenisation natively with
        its CTB-trained tokenizer.
        """
        return text

    def stanza_pipeline_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments used when instantiating the Stanza
        pipeline. English default: standard tokenize → pos →
        constituency pipeline. Chinese subclasses override to point at
        the ``zh-hans`` model."""
        return {
            "lang": "en",
            "processors": "tokenize,pos,constituency",
            "tokenize_no_ssplit": False,
            "verbose": False,
        }

    # split conventions -------------------------------------------------
    def predefined_split(
        self, df: pd.DataFrame
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return a dataset-provided (train_idx, dev_idx, test_idx)
        triple, or None if the dataset has no canonical split.

        Default is None — most datasets (EmoBank, FB, CVAT) don't
        publish a fixed split and the pipeline generates its own via
        k-fold CV. SST overrides this to honor its canonical
        train/dev/test column, which is the standard evaluation
        protocol for that dataset.

        When a processor returns a non-None value, ``--compare``
        dispatches through ``run_comparison_predefined_split`` (single
        training run per architecture, one MWU test) instead of the
        k-fold ``run_comparison`` path.
        """
        return None

    # path conventions --------------------------------------------------
    def fold_checkpoint_path(
        self,
        model_dir: Path,
        dimension: Optional[str],
        model_tag: Optional[str],
        fold: Optional[int],
        region_depth: int,
        run_tag: Optional[str] = None,
    ) -> Path:
        """Per-fold checkpoint path. Used by ModelCheckpoint during
        training; one file is written per (architecture, fold).

        Default convention:
            {model_dir}/{dataset}/[{run_tag}/]{dimension}/{model_tag}/
                best_d{depth}_fold{fold}.keras

        The optional ``run_tag`` folder sits right under the dataset so
        all artifacts from one run are colocated; when None (default)
        the folder is omitted and the layout matches legacy behavior
        exactly. Falls back to legacy flat naming when dimension or
        model_tag is None (e.g. legacy joint-V+A runs), with the
        run_tag appended as a filename suffix in that case.
        """
        if dimension and model_tag:
            base = model_dir / self.name
            if run_tag:
                base = base / run_tag
            sub = base / dimension / model_tag
            sub.mkdir(parents=True, exist_ok=True)
            fold_part = f"_fold{fold}" if fold is not None else ""
            return sub / f"best_d{region_depth}{fold_part}.keras"
        # Legacy fallback for non-comparison runs.
        parts = [f"best_{self.name}", f"d{region_depth}"]
        if dimension:
            parts.append(dimension)
        if model_tag:
            parts.append(model_tag)
        if fold is not None:
            parts.append(f"fold{fold}")
        if run_tag:
            parts.append(run_tag)
        return model_dir / f"{'_'.join(parts)}.keras"

    def best_model_path(
        self,
        model_dir: Path,
        dimension: str,
        model_tag: str,
        region_depth: int,
        run_tag: Optional[str] = None,
    ) -> Path:
        """Stable path for the "best-across-folds" model for a given
        (dataset, dimension, architecture) triple. Used after all folds
        have been trained to copy the winning fold's checkpoint to a
        canonical location.

        Default convention:
            {model_dir}/{dataset}/[{run_tag}/]{dimension}/{model_tag}/
                best_d{depth}_overall.keras

        Sits alongside the per-fold files in the same folder so
        provenance is obvious — the "overall" suffix marks it as the
        cross-fold pick. The optional ``run_tag`` folder keeps runs
        with different configurations from colliding.
        """
        base = model_dir / self.name
        if run_tag:
            base = base / run_tag
        sub = base / dimension / model_tag
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"best_d{region_depth}_overall.keras"


class EmoBankProcessor(BaseDatasetProcessor):
    """Generic processor for any EmoBank-style CSV with columns:
        id, text, V, A, D, stdV, stdA, stdD, N

    The writer-perspective and reader-perspective EmoBank releases
    share this schema, so we consolidate their handling here. Concrete
    subclasses set a fixed ``name`` (used for cache / model / log
    paths) and default CSV filename.
    """

    default_filename: str = "emobank.csv"

    def __init__(
        self,
        filepath: Optional[Path] = None,
        name: Optional[str] = None,
    ) -> None:
        # Subclasses provide a sensible default name; allow override
        # at construction for ad-hoc runs against arbitrary CSVs.
        effective_name = name or getattr(self, "default_name", "emobank")
        super().__init__(filepath if filepath is not None else Path("."), effective_name)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath)
        df = df.dropna(subset=["text"]).reset_index(drop=True)
        df = df[df["text"].astype(str).str.strip().astype(bool)]
        df = df.reset_index(drop=True)
        # Drop rows with non-numeric V/A just in case
        df = df.dropna(subset=["V", "A"]).reset_index(drop=True)
        log.info("[%s] loaded %d rows", self.name, len(df))
        return df

    def get_texts(self, df: pd.DataFrame) -> List[str]:
        return df["text"].astype(str).tolist()

    def get_labels(self, df: pd.DataFrame) -> np.ndarray:
        return df[["V", "A"]].to_numpy(dtype=np.float32)


class EmoBankWriterProcessor(EmoBankProcessor):
    """Writer-perspective EmoBank (texts scored for how the writer
    intended them to feel)."""

    default_name: str = "emobank_writer"
    default_filename: str = "writer_10240.csv"

    def __init__(self, filepath: Path, name: str = "emobank_writer") -> None:
        super().__init__(filepath=filepath, name=name)


class EmoBankReaderProcessor(EmoBankProcessor):
    """Reader-perspective EmoBank (same texts, but scored for how a
    reader perceives them). Same CSV schema as the writer release."""

    default_name: str = "emobank_reader"
    default_filename: str = "reader_10240.csv"

    def __init__(self, filepath: Path, name: str = "emobank_reader") -> None:
        super().__init__(filepath=filepath, name=name)


class FBProcessor(BaseDatasetProcessor):
    """Processor for the Facebook (FB) VA dataset.

    Schema (3 columns, no ID, no dominance, no annotator stats):
        text, V, A

    Notable differences from EmoBank:
      * Label range is approximately 1–9 (vs EmoBank's 1–5); std of V
        and A are ~1.2 and ~2.0 respectively. This affects raw MAE
        scale but not Pearson r (scale-invariant). Per-fold label
        normalization (cfg.normalize_labels=True) keeps training
        dynamics independent of the scale.
      * Texts are longer on average (≈16 words, p99 ≈72) than
        EmoBank's short sentences; the auto-shape mechanism handles
        this at region-tensor construction time.
      * Dataset is smaller (~2,900 rows vs ~10,200); each fold's test
        set is ~290 samples so pooled MWU/Wilcoxon operate on
        n ≈ 1,447 per model.
    """

    default_name: str = "fb"
    default_filename: str = "fb_dataset_cleaned.csv"

    def __init__(self, filepath: Path, name: str = "fb") -> None:
        super().__init__(filepath, name)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath)
        df = df.dropna(subset=["text"]).reset_index(drop=True)
        df = df[df["text"].astype(str).str.strip().astype(bool)]
        df = df.reset_index(drop=True)
        df = df.dropna(subset=["V", "A"]).reset_index(drop=True)
        log.info("[%s] loaded %d rows", self.name, len(df))
        return df

    def get_texts(self, df: pd.DataFrame) -> List[str]:
        return df["text"].astype(str).tolist()

    def get_labels(self, df: pd.DataFrame) -> np.ndarray:
        return df[["V", "A"]].to_numpy(dtype=np.float32)


class CVATProcessor(BaseDatasetProcessor):
    """Processor for the Chinese Valence-Arousal Text (CVAT) dataset.

    Schema (5 columns):
        No., Text, Category, Valence, Arousal

    CVAT-specific handling:
      * **Encoding:** Source CSV is ``big5hkscs`` (Big5 with Hong Kong
        Supplementary Character Set). Plain ``big5`` fails on a handful
        of rows.
      * **Script:** Text is Traditional Chinese. We convert to
        Simplified via OpenCC (``t2s`` config) so downstream resources
        (Stanza zh-hans, Chinese-Word-Vectors trained on zhwiki which
        is predominantly Simplified) align on script.
      * **Tokenisation:** Stanza's Chinese tokenizer (trained jointly
        with the constituency parser on CTB 5.1) handles segmentation
        natively. No separate tokenizer (jieba, etc.) — this keeps
        tokenizer and parser in agreement, avoiding boundary
        mismatches that a separate segmenter would introduce.
      * **Parser:** Stanza's ``zh-hans`` constituency model (trained
        on Chinese Treebank 5.1). The existing region-extraction logic
        works unchanged since it operates on tree topology, not on
        tag names.
      * **Embeddings:** Chinese-Word-Vectors ``sgns.merge.word`` at
        300d by default — the "Mixed-large" release trained by SGNS
        on a merged corpus (Baidu Encyclopedia + zhwiki + People's
        Daily News + Sogou News + Financial News + Zhihu_QA + Weibo
        + Literature). This is a superset of the zhwiki corpus the
        paper cites; the merged release is used here because the
        zhwiki-only file is only distributed through Baidu Netdisk
        (Chinese mobile-number verification required) while the
        merged release is on Google Drive. Broader corpus coverage
        is additionally well-matched to CVAT's mix of news, product,
        and review text. The user can swap via ``--embedding-path``;
        file format is GloVe-compatible (space-separated), with an
        optional vocab-size/dim header line that ``load_glove_matrix``
        naturally skips.

    Dataset stats: ~2,009 rows, V μ≈4.83 σ≈1.37, A μ≈5.05 σ≈0.95,
    labels on approximately a 1-9 scale. 6 categories (news, political,
    hotel, book, car, laptop). Optimal region depth is 8 for both V
    and A (deeper than English datasets because CTB constituency trees
    are denser than PTB trees for comparable sentence lengths).
    """

    default_name: str = "cvat"
    default_filename: str = "CVAT.csv"
    # User downloads from https://github.com/Embedding/Chinese-Word-Vectors
    # (Mixed-large / "综合", Word column, 300d, Google Drive). The file
    # unpacks to "sgns.merge.word" or similar; place at
    # cfg.base_dir / <this filename>. If your filename differs, either
    # rename it or pass --embedding-path at runtime.
    default_embedding_filename: str = "sgns.merge.word"

    language: str = "zh"

    def __init__(self, filepath: Path, name: str = "cvat") -> None:
        super().__init__(filepath, name)
        # Lazy OpenCC converter; resolved on first preprocess so
        # non-CVAT runs don't need opencc installed.
        self._opencc = None

    # --- data loading ------------------------------------------------
    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath, encoding="big5hkscs")
        # Preserve numeric label columns; drop anything without text/labels.
        df = df.dropna(subset=["Text"]).reset_index(drop=True)
        df = df[df["Text"].astype(str).str.strip().astype(bool)]
        df = df.reset_index(drop=True)
        df = df.dropna(subset=["Valence", "Arousal"]).reset_index(drop=True)
        log.info("[%s] loaded %d rows", self.name, len(df))
        return df

    def get_texts(self, df: pd.DataFrame) -> List[str]:
        return df["Text"].astype(str).tolist()

    def get_labels(self, df: pd.DataFrame) -> np.ndarray:
        return df[["Valence", "Arousal"]].to_numpy(dtype=np.float32)

    # --- Chinese-specific preprocessing ------------------------------
    def _ensure_opencc(self):
        if self._opencc is not None:
            return self._opencc
        try:
            from opencc import OpenCC
        except ImportError as exc:
            raise SystemExit(
                "CVAT requires opencc for Traditional->Simplified "
                "conversion. Install with:\n"
                "    pip install opencc-python-reimplemented"
            ) from exc
        # t2s = Traditional to Simplified. Lossless for sentiment text;
        # a few obscure Traditional-only characters have no clean
        # Simplified equivalent but are not common in CVAT.
        self._opencc = OpenCC("t2s")
        return self._opencc

    def prepare_for_parser(self, text: str) -> str:
        """Convert Traditional→Simplified so the Simplified-trained
        Stanza zh-hans pipeline and Simplified-heavy Chinese-Word-Vectors
        both see the script they were trained on. No tokenisation is
        performed here — Stanza segments the text itself, keeping
        tokenizer and parser in agreement.
        """
        opencc = self._ensure_opencc()
        return opencc.convert(text)

    # --- Stanza pipeline config --------------------------------------
    def stanza_pipeline_kwargs(self) -> Dict[str, Any]:
        return {
            # zh-hans = Simplified Chinese. CVAT is converted to
            # Simplified before reaching the parser, so this lines up.
            "lang": "zh-hans",
            "processors": "tokenize,pos,constituency",
            # Let Stanza do its own tokenization and sentence
            # segmentation. Its tokenizer is trained jointly with the
            # constituency parser, so boundaries are consistent.
            "tokenize_no_ssplit": False,
            "verbose": False,
        }


class SSTProcessor(BaseDatasetProcessor):
    """Processor for the Stanford Sentiment Treebank (SST) dataset.

    Schema (3 columns):
        text, label, split

    SST-specific handling:
      * **Canonical split:** The CSV includes a ``split`` column with
        values in {train, dev, test}. This is the standard SST
        evaluation protocol and MUST be respected rather than
        generating a random split. Returning a ``predefined_split``
        causes ``--compare`` to dispatch through
        ``run_comparison_predefined_split``, which trains one LSTM +
        one BiLSTM on this fixed partition and runs MWU on the test
        predictions (no cross-validation).
      * **Valence only:** SST has a single continuous sentiment label
        on [0, 1]; there is no arousal. ``get_labels`` returns an
        (N, 2) array with NaN in the arousal column so the shape
        contract with the rest of the pipeline is preserved; the
        valence column is the only one actually used. ``--dimension
        arousal`` is rejected with a clear error when the dataset is
        SST.
      * **Embeddings/parser:** English — GloVe 840B 300d and Stanza
        English constituency parser, same as EmoBank/FB.

    Dataset stats: 11,855 rows total (train=8,544; dev=1,101; test=2,210),
    label range [0, 1] (μ≈0.51 σ≈0.25), text length ≈17 words mean.
    Optimal region depth is 5 for valence.
    """

    default_name: str = "sst"
    default_filename: str = "stanford_sentiment_assembled.csv"
    language: str = "en"
    # Inherits default_embedding_filename from BaseDatasetProcessor
    # (glove.840B.300d.txt).

    def __init__(self, filepath: Path, name: str = "sst") -> None:
        super().__init__(filepath, name)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath)
        df = df.dropna(subset=["text", "label", "split"]).reset_index(drop=True)
        df = df[df["text"].astype(str).str.strip().astype(bool)]
        df = df.reset_index(drop=True)
        # Normalize split labels to lowercase in case of mixed casing.
        df["split"] = df["split"].astype(str).str.lower().str.strip()
        valid_splits = {"train", "dev", "test"}
        bad = set(df["split"].unique()) - valid_splits
        if bad:
            raise ValueError(
                f"[sst] unknown split values {bad}; expected subset of "
                f"{valid_splits}"
            )
        log.info("[%s] loaded %d rows", self.name, len(df))
        for s in ("train", "dev", "test"):
            log.info("[%s]   split=%s: %d rows", self.name, s, (df["split"] == s).sum())
        return df

    def get_texts(self, df: pd.DataFrame) -> List[str]:
        return df["text"].astype(str).tolist()

    def get_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Return (N, 2) with valence in column 0 and NaN arousal in
        column 1. The NaN column signals "this dataset has no arousal
        label"; any code path that tries to read it will surface the
        issue immediately rather than silently training on zeros."""
        y = np.full((len(df), 2), np.nan, dtype=np.float32)
        y[:, 0] = df["label"].to_numpy(dtype=np.float32)
        return y

    def predefined_split(
        self, df: pd.DataFrame
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return (train_idx, dev_idx, test_idx) from the 'split'
        column. Indices are into the DataFrame as returned by
        ``load()`` (i.e. after NaN-drop / reset_index)."""
        split = df["split"].to_numpy()
        train_idx = np.where(split == "train")[0]
        dev_idx = np.where(split == "dev")[0]
        test_idx = np.where(split == "test")[0]
        return train_idx, dev_idx, test_idx


# ===========================================================================
# 3. Cache utilities
# ===========================================================================
def load_cache(path: Path) -> Optional[Any]:
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            log.warning("cache at %s is unreadable (%s); ignoring", path, exc)
    return None


def save_cache(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


# ===========================================================================
# 4. Constituency parsing with caching
# ===========================================================================
class TreeParser:
    """Thin wrapper around Stanza's constituency pipeline.

    Stanza's constituency parser is GPU-friendly but slow on CPU. Because
    parses are deterministic per text, we cache the full list of trees
    keyed on the dataset name and the text count.

    The parser is driven by the dataset processor so language / pipeline
    configuration lives on the processor (where it belongs), not here.
    """

    def __init__(self, cache_dir: Path, processor: BaseDatasetProcessor) -> None:
        self.cache_dir = cache_dir
        self.processor = processor
        self.dataset_name = processor.name
        self.cache_file = cache_dir / f"parsed_trees_{processor.name}.pkl"
        self._nlp = None

    def _ensure_pipeline(self) -> None:
        if self._nlp is not None:
            return
        import stanza  # local import keeps start-up fast when cache hits
        kwargs = self.processor.stanza_pipeline_kwargs()
        lang = kwargs.get("lang", "en")
        procs = kwargs.get("processors", "tokenize,pos,constituency")
        try:
            self._nlp = stanza.Pipeline(**kwargs)
        except Exception:  # model not downloaded yet
            log.info(
                "downloading Stanza constituency model (lang=%s, processors=%s) ...",
                lang, procs,
            )
            stanza.download(lang, processors=procs, verbose=False)
            self._nlp = stanza.Pipeline(**kwargs)

    def parse_all(self, texts: List[str]) -> List[List[Any]]:
        """Return a list (len == ``len(texts)``) of lists of Stanza constituency trees
        (one tree per sentence per text).

        Each text is passed through ``processor.prepare_for_parser``
        first — the default is identity, but e.g. the Chinese CVAT
        processor uses this hook to do Traditional→Simplified
        conversion via OpenCC before Stanza tokenises and parses.
        """
        cached = load_cache(self.cache_file)
        if cached is not None and isinstance(cached, dict) and cached.get("n") == len(texts):
            log.info("hit parse-tree cache: %s (%d texts)", self.cache_file, len(texts))
            return cached["trees"]

        self._ensure_pipeline()
        trees: List[List[Any]] = []
        t0 = time.time()
        log.info(
            "parsing %d texts with Stanza (lang=%s) ...",
            len(texts), self.processor.language,
        )
        for i, text in enumerate(texts):
            try:
                prepared = self.processor.prepare_for_parser(text)
                doc = self._nlp(prepared)
                sent_trees = [sent.constituency for sent in doc.sentences]
            except Exception as exc:  # noqa: BLE001
                log.warning("parse failed for idx=%d (%s); storing empty tree list", i, exc)
                sent_trees = []
            trees.append(sent_trees)
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                log.info(
                    "  parsed %d/%d (%.1fs, %.1fms/text)",
                    i + 1, len(texts), el, el / (i + 1) * 1000,
                )
        save_cache({"n": len(texts), "trees": trees}, self.cache_file)
        log.info("parsing done in %.1fs; cached to %s", time.time() - t0, self.cache_file)
        return trees


# ===========================================================================
# 5. Region extraction (tree → list of word regions at a given depth)
# ===========================================================================
def _tree_leaves(node: Any) -> List[str]:
    """Return the terminal word labels under a Stanza constituency subtree."""
    children = getattr(node, "children", None) or ()
    if not children:
        return [node.label] if hasattr(node, "label") else []
    out: List[str] = []
    for child in children:
        out.extend(_tree_leaves(child))
    return out


def _all_children_terminal(node: Any) -> bool:
    """True if all children are either leaves or pre-terminals (POS → word)."""
    children = getattr(node, "children", None) or ()
    if not children:
        return True
    for c in children:
        cc = getattr(c, "children", None) or ()
        if not cc:
            continue  # leaf
        # pre-terminal: exactly one child that itself has no children
        if len(cc) == 1 and not (getattr(cc[0], "children", None) or ()):
            continue
        return False
    return True


def extract_regions_at_depth(tree: Any, target_depth: int) -> List[List[str]]:
    """Divide a parse tree into word regions at ``target_depth``.

    The root counts as depth 1. Following the paper's Fig. 3:
      * depth 1 → the whole text is one region
      * increasing depth → regions shrink, capturing phrases and clauses
      * maximum depth → each region is a single word

    To survive real-world parser output where the top has a unary ROOT
    wrapper and many chains have only one internal child, this
    implementation collapses unary chains, which matches the paper's
    schematic trees (Fig. 3 draws binarised / branching trees only).
    When a branch becomes a pre-terminal-only constituent before the
    target depth is reached, its leaves are emitted as a single region
    (the branch is "shorter than target_depth").
    """
    regions: List[List[str]] = []

    def recurse(node: Any, depth: int) -> None:
        children = getattr(node, "children", None) or ()

        # Leaf word — emit as single-word region.
        if not children:
            if hasattr(node, "label"):
                regions.append([node.label])
            return

        # Unary collapse: if this node has exactly one internal child,
        # descend without consuming a depth level.
        if len(children) == 1 and (getattr(children[0], "children", None) or ()):
            recurse(children[0], depth)
            return

        # If every child is a leaf / pre-terminal, we cannot split further
        # meaningfully — emit the whole constituent as one region.
        if _all_children_terminal(node):
            leaves = _tree_leaves(node)
            if leaves:
                regions.append(leaves)
            return

        # Reached target depth — emit the whole constituent as one region.
        if depth >= target_depth:
            leaves = _tree_leaves(node)
            if leaves:
                regions.append(leaves)
            return

        # Otherwise keep descending.
        for c in children:
            if getattr(c, "children", None) or ():
                recurse(c, depth + 1)
            elif hasattr(c, "label"):
                # Stray terminal at this depth — emit singly.
                regions.append([c.label])

    recurse(tree, 1)
    return regions


def text_trees_to_regions(
    sentence_trees: List[Any], target_depth: int
) -> List[List[str]]:
    """Flatten all sentences' regions into one list-of-word-lists per text."""
    regions: List[List[str]] = []
    for tree in sentence_trees:
        regions.extend(extract_regions_at_depth(tree, target_depth))
    if not regions:
        regions.append(["<UNK>"])  # never return zero regions for a text
    return regions


# ===========================================================================
# 6. Vocabulary + GloVe embedding matrix (cached)
# ===========================================================================
def _normalize_token(word: str, case_sensitive: bool) -> str:
    return word if case_sensitive else word.lower()


def build_vocab(train_regions: List[List[List[str]]], cfg: Config) -> Dict[str, int]:
    """Build vocabulary indices from training regions only.

    Index 0 is reserved for ``<PAD>`` and index 1 for ``<UNK>``.
    Casing is preserved when ``cfg.case_sensitive`` is True.
    """
    vocab: Dict[str, int] = {"<PAD>": cfg.pad_token_id, "<UNK>": cfg.unk_token_id}
    for text in train_regions:
        for region in text:
            for w in region:
                k = _normalize_token(w, cfg.case_sensitive)
                if k not in vocab:
                    vocab[k] = len(vocab)
    return vocab


def load_glove_matrix(
    vocab: Dict[str, int],
    cfg: Config,
    dataset_name: str,
    emb_path: Path,
) -> np.ndarray:
    """Return an (vocab_size, embedding_dim) float32 matrix, caching it per
    (dataset, casing, embedding-file, vocab-content) tuple so fold-specific
    vocabs and language-specific embedding files do not collide.

    Accepts text files in either standard GloVe format (``word v1 v2 ... vN``
    per line, no header) or word2vec .vec format (``vocab_size dim`` header
    line followed by GloVe-like lines). The header of .vec files is
    naturally skipped because it has only 2 whitespace-separated fields,
    while GloVe lines have ``embedding_dim + 1``.
    """
    import hashlib

    cache_suffix = "cs" if cfg.case_sensitive else "ci"
    vocab_fp = hashlib.sha256(
        "\n".join(sorted(vocab.keys())).encode("utf-8")
    ).hexdigest()[:12]
    # Include a short hash of the embedding file path in the cache key so
    # different embedding releases (GloVe vs Chinese-Word-Vectors vs
    # FastText) never share a cache entry.
    emb_fp = hashlib.sha256(str(emb_path.resolve()).encode("utf-8")).hexdigest()[:8]
    cache_path = (
        cfg.cache_dir
        / f"embedding_matrix_{dataset_name}_{cache_suffix}_{emb_fp}_{vocab_fp}.pkl"
    )
    cached = load_cache(cache_path)
    if cached is not None and cached.get("vocab_size") == len(vocab):
        log.info("hit embedding cache: %s", cache_path)
        return cached["matrix"]

    rng = np.random.RandomState(cfg.random_seed)
    matrix = np.zeros((len(vocab), cfg.embedding_dim), dtype=np.float32)
    # Initialise <UNK> with a small random vector; <PAD> stays zero.
    matrix[cfg.unk_token_id] = rng.normal(0.0, 0.01, cfg.embedding_dim).astype(np.float32)

    if not emb_path.exists():
        log.warning(
            "Embedding file missing at %s — falling back to random init "
            "for all tokens.",
            emb_path,
        )
        for i in range(2, len(vocab)):
            matrix[i] = rng.normal(0.0, 0.01, cfg.embedding_dim).astype(np.float32)
        save_cache({"vocab_size": len(vocab), "matrix": matrix}, cache_path)
        return matrix

    vocab_set = set(vocab.keys())
    found = 0
    log.info(
        "streaming embeddings from %s (case_%s) ...",
        emb_path, "sensitive" if cfg.case_sensitive else "insensitive",
    )
    t0 = time.time()
    with open(emb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, 1):
            # GloVe 840B: space-separated token<sp>vec<sp>...<sp>vec.
            # word2vec .vec format: first line is "vocab_size dim" header
            # (only 2 fields, so naturally skipped by the length check).
            # Some tokens contain spaces — rsplit on the last N tokens.
            parts = line.rstrip("\n").rsplit(" ", cfg.embedding_dim)
            if len(parts) != cfg.embedding_dim + 1:
                continue
            word = parts[0]
            key = word if cfg.case_sensitive else word.lower()
            if key in vocab_set:
                try:
                    vec = np.asarray(parts[1:], dtype=np.float32)
                except ValueError:
                    continue
                idx = vocab[key]
                # First match wins in case-insensitive mode; only match wins
                # in case-sensitive mode.
                if not matrix[idx].any():
                    matrix[idx] = vec
                    found += 1
            if line_num % 500_000 == 0:
                log.info("  scanned %dM lines (%.1fs)", line_num // 1_000_000, time.time() - t0)

    # Fill any remaining OOV slots (besides pad) with small random noise.
    rng2 = np.random.RandomState(cfg.random_seed + 1)
    for tok, idx in vocab.items():
        if idx in (cfg.pad_token_id,):
            continue
        if not matrix[idx].any() and idx != cfg.unk_token_id:
            matrix[idx] = rng2.normal(0.0, 0.01, cfg.embedding_dim).astype(np.float32)

    log.info(
        "embedding coverage: %d/%d (%.1f%%) in %.1fs",
        found, len(vocab), 100.0 * found / max(1, len(vocab)), time.time() - t0,
    )
    save_cache({"vocab_size": len(vocab), "matrix": matrix}, cache_path)
    return matrix


# ===========================================================================
# 7. Tensor conversion
# ===========================================================================
def regions_to_tensor(
    all_text_regions: List[List[List[str]]],
    vocab: Dict[str, int],
    cfg: Config,
) -> np.ndarray:
    """Convert nested regions into an int32 array of shape (num_texts, R, N).

    Tokens beyond R regions or N words per region are truncated.
    Missing slots are zero-padded.
    """
    R, N = cfg.max_num_regions, cfg.max_region_len
    out = np.zeros((len(all_text_regions), R, N), dtype=np.int32)
    unk = cfg.unk_token_id
    for i, text_regions in enumerate(all_text_regions):
        for r, region in enumerate(text_regions[:R]):
            for w_idx, word in enumerate(region[:N]):
                key = _normalize_token(word, cfg.case_sensitive)
                out[i, r, w_idx] = vocab.get(key, unk)
    return out


def maybe_auto_shape(
    all_text_regions: List[List[List[str]]],
    cfg: Config,
) -> None:
    """Set cfg.max_num_regions / max_region_len to the given percentile of data."""
    if not cfg.auto_shape:
        return
    percentile = cfg.auto_shape_percentile
    region_lens = [len(r) for text in all_text_regions for r in text]
    region_counts = [len(text) for text in all_text_regions]
    if region_lens:
        cfg.max_region_len = max(
            cfg.cnn_kernel_size + 1, int(np.percentile(region_lens, percentile))
        )
    if region_counts:
        cfg.max_num_regions = max(1, int(np.percentile(region_counts, percentile)))
    log.info(
        "auto-shape @ p%.1f → max_num_regions=R=%d, max_region_len=N=%d",
        percentile, cfg.max_num_regions, cfg.max_region_len,
    )


# ===========================================================================
# 8. Model
# ===========================================================================
def _ccc_loss(y_true, y_pred):
    """Concordance Correlation Coefficient loss (1 - mean CCC over dims).

    CCC penalizes low prediction variance and mean-shift directly; it is
    the standard loss for VA regression where MSE-trained models tend to
    collapse predictions to the mean.
    """
    import tensorflow as tf

    # Treat each batch as a single population over axis 0 (samples).
    mu_pred = tf.reduce_mean(y_pred, axis=0, keepdims=True)
    mu_true = tf.reduce_mean(y_true, axis=0, keepdims=True)
    var_pred = tf.reduce_mean(tf.square(y_pred - mu_pred), axis=0)
    var_true = tf.reduce_mean(tf.square(y_true - mu_true), axis=0)
    cov = tf.reduce_mean((y_pred - mu_pred) * (y_true - mu_true), axis=0)

    denom = var_pred + var_true + tf.square(mu_pred - mu_true)[0] + 1e-8
    ccc = 2.0 * cov / denom
    return 1.0 - tf.reduce_mean(ccc)


def build_model(vocab_size: int, embedding_matrix: np.ndarray, cfg: Config):
    """Construct the tree-structured regional CNN-LSTM model.

    Shape journey:
        input                : (B, R, N)  int32  word IDs
        embedding            : (B, R, N, E)
        TD SpatialDropout1D  : (B, R, N, E)  word-vector dropout
        TD Conv1D            : (B, R, N-k+1, F)
        TD SpatialDropout1D  : (B, R, N-k+1, F)  CNN output dropout
        TD MaxPool1D         : (B, R, (N-k+1)//p, F)
        TD Flatten           : (B, R, flat)
        padded-region mask   : (B, R, flat)  zero rows for pad regions
        Masking              : propagates mask to LSTM
        LSTM (or BiLSTM)     : (B, H) or (B, 2H)  text vector t
        Dropout              : (B, H)
        Dense(2, linear)     : (B, 2)       [valence, arousal]
    """
    import tensorflow as tf
    from tensorflow.keras import Model, layers, regularizers

    reg = regularizers.l2(cfg.l2_reg) if cfg.l2_reg > 0 else None

    inputs = layers.Input(
        shape=(cfg.max_num_regions, cfg.max_region_len), dtype="int32", name="tokens"
    )

    # Regions are valid if they contain at least one non-PAD token.
    region_valid = layers.Lambda(
        lambda x: tf.reduce_any(tf.not_equal(x, 0), axis=-1),
        name="region_valid",
    )(inputs)  # (B, R) bool

    emb = layers.Embedding(
        input_dim=vocab_size,
        output_dim=cfg.embedding_dim,
        weights=[embedding_matrix],
        trainable=cfg.trainable_embeddings,
        mask_zero=False,  # masking handled at the region level
        name="glove_embedding",
    )(inputs)  # (B, R, N, E)

    # --- input-level SpatialDropout1D over word vectors ----------------
    if cfg.embedding_dropout > 0:
        emb = layers.TimeDistributed(
            layers.SpatialDropout1D(cfg.embedding_dropout),
            name="td_emb_dropout",
        )(emb)

    # --- per-region CNN over words ------------------------------------
    # Default (paper): single kernel of width cfg.cnn_kernel_size with
    # cfg.cnn_filters filters.
    # --multi-kernel: parallel branches at each kernel size in
    # cfg.multi_kernel_sizes with cfg.multi_kernel_filters_per_size each;
    # branch outputs are concatenated along the feature axis.
    if cfg.multi_kernel:
        # Safety: every kernel must fit within the region length.
        too_big = [k for k in cfg.multi_kernel_sizes if k > cfg.max_region_len]
        if too_big:
            raise ValueError(
                f"multi_kernel sizes {too_big} exceed max_region_len="
                f"{cfg.max_region_len}; either disable --multi-kernel or "
                f"raise auto_shape_percentile."
            )

        branches = []
        for k in cfg.multi_kernel_sizes:
            b = layers.TimeDistributed(
                layers.Conv1D(
                    filters=cfg.multi_kernel_filters_per_size,
                    kernel_size=k,
                    activation="relu",
                    padding="valid",
                    kernel_regularizer=reg,
                ),
                name=f"td_conv_k{k}",
            )(emb)
            if cfg.spatial_dropout > 0:
                b = layers.TimeDistributed(
                    layers.SpatialDropout1D(cfg.spatial_dropout),
                    name=f"td_spatial_dropout_k{k}",
                )(b)
            b = layers.TimeDistributed(
                layers.MaxPooling1D(pool_size=cfg.pool_size),
                name=f"td_maxpool_k{k}",
            )(b)
            b = layers.TimeDistributed(layers.Flatten(), name=f"td_flatten_k{k}")(b)
            branches.append(b)

        flat = layers.Concatenate(axis=-1, name="td_concat_kernels")(branches)
    else:
        conv = layers.TimeDistributed(
            layers.Conv1D(
                filters=cfg.cnn_filters,
                kernel_size=cfg.cnn_kernel_size,
                activation="relu",
                padding="valid",
                kernel_regularizer=reg,
            ),
            name="td_conv",
        )(emb)
        if cfg.spatial_dropout > 0:
            conv = layers.TimeDistributed(
                layers.SpatialDropout1D(cfg.spatial_dropout),
                name="td_spatial_dropout",
            )(conv)
        pooled = layers.TimeDistributed(
            layers.MaxPooling1D(pool_size=cfg.pool_size), name="td_maxpool"
        )(conv)
        flat = layers.TimeDistributed(layers.Flatten(), name="td_flatten")(pooled)
    # flat: (B, R, flat_dim)

    # --- zero-out padded regions so Masking can drop them -------------
    mask_expanded = layers.Lambda(
        lambda m: tf.cast(m, tf.float32)[..., None], name="mask_expand"
    )(region_valid)
    flat = layers.Multiply(name="apply_region_mask")([flat, mask_expanded])
    flat = layers.Masking(mask_value=0.0, name="region_masking")(flat)

    # --- sequential layer across regions ------------------------------
    lstm_layer = layers.LSTM(
        units=cfg.lstm_units,
        recurrent_dropout=cfg.recurrent_dropout,
        return_sequences=False,
        kernel_regularizer=reg,
        name="region_lstm",
    )
    if cfg.bidirectional_lstm:
        text_vec = layers.Bidirectional(lstm_layer, name="region_bilstm")(flat)
    else:
        text_vec = lstm_layer(flat)

    # --- dropout before the linear decoder ----------------------------
    if cfg.post_lstm_dropout > 0:
        text_vec = layers.Dropout(cfg.post_lstm_dropout, name="post_lstm_dropout")(
            text_vec
        )

    # --- linear decoder ------------------------------------------------
    output_dim = 1 if cfg.target_dimension in ("valence", "arousal") else 2
    decoder_name = (
        f"{cfg.target_dimension[0]}_linear_decoder"
        if cfg.target_dimension in ("valence", "arousal")
        else "va_linear_decoder"
    )
    outputs = layers.Dense(
        output_dim, activation="linear", kernel_regularizer=reg, name=decoder_name
    )(text_vec)

    model = Model(inputs, outputs, name="TreeRegionalCNNLSTM")

    optimizer_kwargs = {"learning_rate": cfg.learning_rate}
    if cfg.grad_clip_norm > 0:
        optimizer_kwargs["clipnorm"] = cfg.grad_clip_norm

    if cfg.loss_type == "ccc":
        loss_fn = _ccc_loss
    elif cfg.loss_type == "mse":
        loss_fn = "mse"
    else:
        raise ValueError(f"unknown loss_type: {cfg.loss_type!r}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(**optimizer_kwargs),
        loss=loss_fn,
        metrics=["mae"],
    )
    return model


# ===========================================================================
# 9. Metrics
# ===========================================================================
def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def evaluate_va(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for idx, dim in enumerate(["valence", "arousal"]):
        metrics[f"{dim}_r"] = pearson_r(y_true[:, idx], y_pred[:, idx])
        metrics[f"{dim}_MAE"] = mae(y_true[:, idx], y_pred[:, idx])
    return metrics


def evaluate_single_dim(
    y_true: np.ndarray, y_pred: np.ndarray, dim_name: str
) -> Dict[str, float]:
    """Single-dimension evaluation. Inputs are (N,) or (N, 1); the
    dim_name must be 'valence' or 'arousal' and becomes the metric key
    prefix.
    """
    yt = np.asarray(y_true, dtype=np.float64).flatten()
    yp = np.asarray(y_pred, dtype=np.float64).flatten()
    return {
        f"{dim_name}_r": pearson_r(yt, yp),
        f"{dim_name}_MAE": mae(yt, yp),
    }

# ===========================================================================
# 10. Pipeline orchestration
# ===========================================================================
def split_indices(n: int, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(cfg.random_seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(cfg.train_ratio * n)
    n_val = int(cfg.val_ratio * n)
    return (
        idx[:n_train],
        idx[n_train : n_train + n_val],
        idx[n_train + n_val :],
    )


def run(processor: BaseDatasetProcessor, cfg: Config) -> Dict[str, Any]:
    """End-to-end training + evaluation with an internally computed
    70/20/10 random split (controlled by ``cfg.random_seed``).

    For real cross-validation use ``run_kfold`` instead.
    """
    # Ensure output directories exist.
    for d in (cfg.cache_dir, cfg.model_dir, cfg.logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # (a) Load + parse --------------------------------------------------
    df = processor.load()
    texts = processor.get_texts(df)
    y = processor.get_labels(df)

    parser = TreeParser(cfg.cache_dir, processor)
    trees = parser.parse_all(texts)

    # (b) Region extraction --------------------------------------------
    log.info("extracting regions at depth=%d ...", cfg.region_depth)
    all_regions = [text_trees_to_regions(t, cfg.region_depth) for t in trees]

    region_counts = np.array([len(r) for r in all_regions])
    region_lens = np.array([len(r) for text in all_regions for r in text])
    log.info(
        "regions/text: mean=%.2f, max=%d | tokens/region: mean=%.2f, max=%d",
        region_counts.mean(), region_counts.max(),
        region_lens.mean() if len(region_lens) else 0.0,
        region_lens.max() if len(region_lens) else 0,
    )
    maybe_auto_shape(all_regions, cfg)

    # (c) Splits --------------------------------------------------------
    train_idx, val_idx, test_idx = split_indices(len(texts), cfg)
    log.info(
        "splits: train=%d, val=%d, test=%d",
        len(train_idx), len(val_idx), len(test_idx),
    )

    report, _, _ = _run_core(
        processor, cfg, texts, y, all_regions,
        train_idx, val_idx, test_idx,
        write_metrics_file=True,
    )
    return report


def _run_core(
    processor: BaseDatasetProcessor,
    cfg: Config,
    texts: List[str],
    y: np.ndarray,
    all_regions: List[List[List[str]]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    write_metrics_file: bool = True,
    fold_tag: Optional[str] = None,
    model_tag: Optional[str] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Train + evaluate a single model on a specific split. Returns the
    metrics report, the original-scale test labels, and the original-scale
    test predictions (so callers like ``run_kfold`` can compute pooled r).

    Operating shapes depend on ``cfg.target_dimension``:
      * None                   → (N, 2) labels, joint V+A training
      * "valence" | "arousal"  → (N, 1) labels, single-dim training
    """
    # Single-dim slicing: do it up front so all downstream shapes match.
    dim_index_map = {"valence": 0, "arousal": 1}
    single_dim = cfg.target_dimension in dim_index_map
    if single_dim:
        dim_name = cfg.target_dimension
        dim_idx = dim_index_map[dim_name]
        y_used = y[:, dim_idx:dim_idx + 1]  # keep (N, 1) shape
        log.info(
            "single-dim mode: target=%s (column %d), depth=%d",
            dim_name, dim_idx, cfg.region_depth,
        )
    else:
        dim_name = None  # noqa: F841 (defensive; only used under single_dim)
        y_used = y

    # (d) Vocabulary from training fold only ---------------------------
    vocab = build_vocab([all_regions[i] for i in train_idx], cfg)
    log.info("vocabulary size (train-only): %d", len(vocab))

    # (e) Embedding matrix (cache keyed on vocab size, so each fold's
    # vocab gets its own cache entry) ---------------------------------
    emb = load_glove_matrix(vocab, cfg, processor.name, processor.embedding_path(cfg))

    # (f) Tensors -------------------------------------------------------
    X = regions_to_tensor(all_regions, vocab, cfg)
    X_train, y_train = X[train_idx], y_used[train_idx]
    X_val, y_val = X[val_idx], y_used[val_idx]
    X_test, y_test = X[test_idx], y_used[test_idx]

    # (f.1) Optional label standardization (per training fold) --------
    y_mean: np.ndarray
    y_std: np.ndarray
    if cfg.normalize_labels:
        y_mean = y_train.mean(axis=0)
        y_std = y_train.std(axis=0)
        y_std = np.where(y_std < 1e-6, 1.0, y_std)
        y_train = (y_train - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std
        log.info(
            "label normalization → mean=%s, std=%s",
            np.round(y_mean, 4).tolist(), np.round(y_std, 4).tolist(),
        )
    else:
        y_mean = np.zeros(y_used.shape[1], dtype=np.float32)
        y_std = np.ones(y_used.shape[1], dtype=np.float32)

    # (g) Model ---------------------------------------------------------
    model = build_model(len(vocab), emb, cfg)
    model.summary(print_fn=lambda s: log.info(s))

    # (h) Train ---------------------------------------------------------
    import tensorflow as tf

    tf.keras.utils.set_random_seed(cfg.random_seed)

    # Delegate checkpoint naming to the processor so subclasses can
    # override per-dataset conventions. Parse fold number out of the
    # tag if it follows the "foldN" convention used by run_kfold /
    # run_comparison; otherwise leave it None.
    fold_num: Optional[int] = None
    if fold_tag and fold_tag.startswith("fold"):
        try:
            fold_num = int(fold_tag[4:])
        except ValueError:
            fold_num = None
    ckpt_path = processor.fold_checkpoint_path(
        model_dir=cfg.model_dir,
        dimension=cfg.target_dimension,
        model_tag=model_tag,
        fold=fold_num,
        region_depth=cfg.region_depth,
        run_tag=cfg.run_tag,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_lr,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # (i) Evaluate ------------------------------------------------------
    y_pred = model.predict(X_test, batch_size=cfg.batch_size, verbose=0)
    y_val_pred = model.predict(X_val, batch_size=cfg.batch_size, verbose=0)

    if cfg.normalize_labels:
        y_pred = y_pred * y_std + y_mean
        y_val_pred = y_val_pred * y_std + y_mean
        y_val_original = y_used[val_idx]
    else:
        y_val_original = y_val

    if single_dim:
        metrics = evaluate_single_dim(y_test, y_pred, cfg.target_dimension)
        val_metrics = evaluate_single_dim(
            y_val_original, y_val_pred, cfg.target_dimension
        )
    else:
        metrics = evaluate_va(y_test, y_pred)
        val_metrics = evaluate_va(y_val_original, y_val_pred)

    log.info("=== Prediction distribution vs. ground truth (test) ===")
    if single_dim:
        gt = y_test.flatten()
        pr = y_pred.flatten()
        gt_std = float(gt.std())
        log.info(
            "  %s: ground-truth μ=%.3f σ=%.3f | predicted μ=%.3f σ=%.3f"
            " (σ_pred/σ_true=%.2f)",
            cfg.target_dimension,
            float(gt.mean()), gt_std,
            float(pr.mean()), float(pr.std()),
            pr.std() / gt_std if gt_std else 0.0,
        )
    else:
        for idx, dim in enumerate(["valence", "arousal"]):
            gt_mean = float(y_test[:, idx].mean())
            gt_std = float(y_test[:, idx].std())
            pr_mean = float(y_pred[:, idx].mean())
            pr_std = float(y_pred[:, idx].std())
            log.info(
                "  %s: ground-truth μ=%.3f σ=%.3f | predicted μ=%.3f σ=%.3f"
                " (σ_pred/σ_true=%.2f)",
                dim, gt_mean, gt_std, pr_mean, pr_std,
                pr_std / gt_std if gt_std else 0.0,
            )

    log.info("=== Dev (val) set performance ===")
    for k, v in val_metrics.items():
        log.info("  dev_%s: %.4f", k, v)
    log.info("=== Test set performance ===")
    for k, v in metrics.items():
        log.info("  %s: %.4f", k, v)

    val_losses = history.history["val_loss"]
    best_epoch = int(np.argmin(val_losses)) + 1
    log.info(
        "training: best val_loss=%.4f at epoch %d/%d",
        min(val_losses), best_epoch, len(val_losses),
    )

    report = {
        "dataset": processor.name,
        "region_depth": cfg.region_depth,
        "target_dimension": cfg.target_dimension,
        "model_tag": model_tag,
        "loss_type": cfg.loss_type,
        "bidirectional_lstm": cfg.bidirectional_lstm,
        "multi_kernel": cfg.multi_kernel,
        "multi_kernel_sizes": list(cfg.multi_kernel_sizes) if cfg.multi_kernel else None,
        "multi_kernel_filters_per_size": cfg.multi_kernel_filters_per_size if cfg.multi_kernel else None,
        "trainable_embeddings": cfg.trainable_embeddings,
        "normalize_labels": cfg.normalize_labels,
        "case_sensitive": cfg.case_sensitive,
        "run_tag": cfg.run_tag,
        "auto_shape_percentile": cfg.auto_shape_percentile,
        "vocab_size": len(vocab),
        "max_num_regions": cfg.max_num_regions,
        "max_region_len": cfg.max_region_len,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "best_epoch": best_epoch,
        "total_epochs_trained": len(val_losses),
        "final_val_loss": float(min(val_losses)),
        **{f"dev_{k}": float(v) for k, v in val_metrics.items()},
        **{k: float(v) for k, v in metrics.items()},
    }

    if write_metrics_file:
        name_parts = [f"metrics_{processor.name}", f"d{cfg.region_depth}"]
        if cfg.target_dimension:
            name_parts.append(cfg.target_dimension)
        if model_tag:
            name_parts.append(model_tag)
        if fold_tag:
            name_parts.append(fold_tag)
        if cfg.run_tag:
            name_parts.append(cfg.run_tag)
        filename = "_".join(name_parts)
        metrics_path = cfg.logs_dir / f"{filename}.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log.info("wrote metrics to %s", metrics_path)

    return report, y_test, y_pred


def run_depth_sweep(
    processor: BaseDatasetProcessor,
    cfg: Config,
    depths: List[int],
) -> Dict[str, Any]:
    """Train at each depth in ``depths``, pick the best by dev loss,
    and report the test metrics of the chosen model. This replicates
    the paper's own depth-selection methodology (Section IV-D): "The
    optimal settings of the division depth were determined using the
    development set of all VA datasets."
    """
    import copy as _copy

    log.info("=== starting depth sweep: %s ===", depths)
    all_results: Dict[int, Dict[str, Any]] = {}

    for d in depths:
        log.info("--- depth sweep: training at depth=%d ---", d)
        d_cfg = _copy.deepcopy(cfg)
        d_cfg.region_depth = d
        all_results[d] = run(processor, d_cfg)

    log.info("=== Depth-sweep summary ===")
    log.info(
        "%5s  %10s  %10s  %10s  %10s  %10s",
        "depth", "val_loss", "dev_V_r", "dev_A_r", "test_V_r", "test_A_r",
    )
    for d in sorted(depths):
        r = all_results[d]
        log.info(
            "%5d  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f",
            d,
            r["final_val_loss"],
            r["dev_valence_r"], r["dev_arousal_r"],
            r["valence_r"], r["arousal_r"],
        )

    best_d = min(all_results, key=lambda k: all_results[k]["final_val_loss"])
    log.info(
        "best depth (by val loss): %d → test valence_r=%.4f, arousal_r=%.4f",
        best_d,
        all_results[best_d]["valence_r"],
        all_results[best_d]["arousal_r"],
    )

    # Persist the sweep summary.
    summary = {
        "dataset": processor.name,
        "sweep_depths": depths,
        "best_depth": best_d,
        "run_tag": cfg.run_tag,
        "per_depth": {str(d): all_results[d] for d in depths},
    }
    tag_suffix = f"_{cfg.run_tag}" if cfg.run_tag else ""
    summary_path = cfg.logs_dir / f"depth_sweep_{processor.name}{tag_suffix}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("wrote depth-sweep summary to %s", summary_path)

    return all_results[best_d]


def run_multi_seed(
    processor: BaseDatasetProcessor,
    cfg: Config,
    seeds: List[int],
) -> Dict[str, Any]:
    """Run the pipeline once per seed (each seed is an independent 70/20/10
    split + training run) and report the mean ± std of test metrics.

    This replicates the paper's 5-fold CV methodology: "EmoBank (both
    reader and writer), FB and CVAT were randomly split into training,
    development and test sets using a 7:2:1 ratio for 5-fold
    cross-validation." (Section IV-A). The paper reports the mean.
    """
    import copy as _copy

    log.info("=== starting multi-seed run: seeds=%s ===", seeds)
    per_seed: Dict[int, Dict[str, Any]] = {}

    for seed in seeds:
        log.info("--- multi-seed: training with seed=%d ---", seed)
        s_cfg = _copy.deepcopy(cfg)
        s_cfg.random_seed = seed
        per_seed[seed] = run(processor, s_cfg)

    metric_keys = [
        "valence_r", "valence_MAE", "arousal_r", "arousal_MAE",
        "dev_valence_r", "dev_arousal_r", "final_val_loss",
    ]
    agg: Dict[str, float] = {}
    for k in metric_keys:
        vals = np.array([per_seed[s][k] for s in seeds], dtype=np.float64)
        agg[f"{k}_mean"] = float(vals.mean())
        agg[f"{k}_std"] = float(vals.std(ddof=1) if len(vals) > 1 else 0.0)

    log.info("=== Multi-seed summary over %d seeds ===", len(seeds))
    log.info(
        "%5s  %10s  %10s  %10s  %10s",
        "seed", "val_loss", "dev_V_r", "test_V_r", "test_A_r",
    )
    for s in seeds:
        r = per_seed[s]
        log.info(
            "%5d  %10.4f  %10.4f  %10.4f  %10.4f",
            s, r["final_val_loss"], r["dev_valence_r"],
            r["valence_r"], r["arousal_r"],
        )
    log.info(
        "mean:  %10.4f  %10.4f  %10.4f  %10.4f",
        agg["final_val_loss_mean"], agg["dev_valence_r_mean"],
        agg["valence_r_mean"], agg["arousal_r_mean"],
    )
    log.info(
        "std:   %10.4f  %10.4f  %10.4f  %10.4f",
        agg["final_val_loss_std"], agg["dev_valence_r_std"],
        agg["valence_r_std"], agg["arousal_r_std"],
    )
    log.info(
        "paper-comparable test scores: valence_r = %.4f ± %.4f, "
        "arousal_r = %.4f ± %.4f",
        agg["valence_r_mean"], agg["valence_r_std"],
        agg["arousal_r_mean"], agg["arousal_r_std"],
    )

    summary = {
        "dataset": processor.name,
        "seeds": seeds,
        "region_depth": cfg.region_depth,
        "run_tag": cfg.run_tag,
        "per_seed": {str(s): per_seed[s] for s in seeds},
        **agg,
    }
    tag_suffix = f"_{cfg.run_tag}" if cfg.run_tag else ""
    summary_path = cfg.logs_dir / (
        f"multiseed_{processor.name}_d{cfg.region_depth}{tag_suffix}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("wrote multi-seed summary to %s", summary_path)

    return summary


def _make_kfolds(n: int, num_folds: int, seed: int) -> List[np.ndarray]:
    """Partition ``range(n)`` into ``num_folds`` disjoint folds.

    The first ``n % num_folds`` folds have one extra element. Returns
    a list of numpy arrays of indices (ordered, shuffled once globally).
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    base = n // num_folds
    remainder = n % num_folds
    folds: List[np.ndarray] = []
    start = 0
    for k in range(num_folds):
        size = base + (1 if k < remainder else 0)
        folds.append(perm[start : start + size])
        start += size
    return folds


def run_kfold(
    processor: BaseDatasetProcessor,
    cfg: Config,
    num_folds: int = 5,
) -> Dict[str, Any]:
    """Real k-fold cross-validation with a 70/20/10 train/dev/test ratio
    per fold (matching Wang et al. Section IV-A: "training, development
    and test sets using a 7:2:1 ratio for 5-fold cross-validation").

    How the splits are built:
        * Partition the dataset into 10 equal tenths (NOT K folds).
        * For each fold k in [0, num_folds):
              test  = tenth k                   (10%)
              dev   = tenths (k+1, k+2) mod 10  (20%)
              train = the remaining 7 tenths    (70%)
        * Test tenths do not overlap across folds, matching standard
          CV semantics.

    With ``num_folds=5`` (the paper's choice), the test tenth rotates
    through tenths 0–4 only; tenths 5–9 never serve as test and only
    appear in train/dev. That is how the paper's "5-fold CV at 7:2:1"
    phrase resolves — 5 iterations, each at 70/20/10, with non-overlapping
    test tenths. Setting ``num_folds=10`` instead would test every sample
    exactly once at the cost of 2× compute.
    """
    import copy as _copy

    TENTHS = 10

    if num_folds < 2 or num_folds > TENTHS:
        raise ValueError(
            f"num_folds must be in [2, {TENTHS}]; got {num_folds}. "
            f"(The internal partition is always 10 tenths for a 70/20/10 "
            f"ratio; num_folds controls how many test-tenths to cycle "
            f"through.)"
        )

    # Ensure output directories exist.
    for d in (cfg.cache_dir, cfg.model_dir, cfg.logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    log.info(
        "=== starting %d-fold CV (70/20/10 per fold, 10 internal tenths) ===",
        num_folds,
    )

    # Load + parse + extract regions ONCE — these don't depend on splits.
    df = processor.load()
    texts = processor.get_texts(df)
    y = processor.get_labels(df)

    parser = TreeParser(cfg.cache_dir, processor)
    trees = parser.parse_all(texts)

    log.info("extracting regions at depth=%d ...", cfg.region_depth)
    all_regions = [text_trees_to_regions(t, cfg.region_depth) for t in trees]

    region_counts = np.array([len(r) for r in all_regions])
    region_lens = np.array([len(r) for text in all_regions for r in text])
    log.info(
        "regions/text: mean=%.2f, max=%d | tokens/region: mean=%.2f, max=%d",
        region_counts.mean(), region_counts.max(),
        region_lens.mean() if len(region_lens) else 0.0,
        region_lens.max() if len(region_lens) else 0,
    )
    maybe_auto_shape(all_regions, cfg)

    # Partition into TENTHS non-overlapping tenths (10% each).
    tenths = _make_kfolds(len(texts), TENTHS, cfg.random_seed)
    for i, t in enumerate(tenths):
        log.info("  tenth %d: %d samples", i, len(t))

    per_fold: List[Dict[str, Any]] = []
    pooled_y: List[np.ndarray] = []
    pooled_pred: List[np.ndarray] = []

    for k in range(num_folds):
        log.info("")
        log.info("=" * 70)
        log.info("--- fold %d/%d ---", k + 1, num_folds)
        log.info("=" * 70)

        test_idx = tenths[k]
        dev_tenth_a = (k + 1) % TENTHS
        dev_tenth_b = (k + 2) % TENTHS
        dev_idx = np.concatenate([tenths[dev_tenth_a], tenths[dev_tenth_b]])
        train_tenths = [
            tenths[i]
            for i in range(TENTHS)
            if i not in (k, dev_tenth_a, dev_tenth_b)
        ]
        train_idx = np.concatenate(train_tenths)

        log.info(
            "splits: train=%d (%.1f%%), dev=%d (%.1f%%), test=%d (%.1f%%)",
            len(train_idx), 100.0 * len(train_idx) / len(texts),
            len(dev_idx), 100.0 * len(dev_idx) / len(texts),
            len(test_idx), 100.0 * len(test_idx) / len(texts),
        )
        log.info(
            "  test=tenth %d, dev=tenths %d+%d, train=remaining 7 tenths",
            k, dev_tenth_a, dev_tenth_b,
        )

        fold_cfg = _copy.deepcopy(cfg)
        # Vary the seed so each fold's model init is different; the data
        # splits are already deterministic from the global partition.
        fold_cfg.random_seed = cfg.random_seed + k

        report, y_test, y_pred = _run_core(
            processor, fold_cfg, texts, y, all_regions,
            train_idx, dev_idx, test_idx,
            write_metrics_file=True,
            fold_tag=f"fold{k + 1}",
        )
        report["fold"] = k + 1
        per_fold.append(report)
        pooled_y.append(y_test)
        pooled_pred.append(y_pred)

    # --- aggregate ----------------------------------------------------
    metric_keys = [
        "valence_r", "valence_MAE",
        "arousal_r", "arousal_MAE",
        "dev_valence_r", "dev_valence_MAE",
        "dev_arousal_r", "dev_arousal_MAE",
        "final_val_loss",
    ]
    agg: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        values = np.array([f[key] for f in per_fold], dtype=np.float64)
        agg[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "values": [float(v) for v in values],
        }

    # Pooled metrics — concatenate predictions over all tested folds.
    pooled_y_arr = np.concatenate(pooled_y, axis=0)
    pooled_pred_arr = np.concatenate(pooled_pred, axis=0)
    pooled_metrics = evaluate_va(pooled_y_arr, pooled_pred_arr)

    # --- report -------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info("=== %d-fold CV summary (70/20/10 per fold) ===", num_folds)
    log.info("=" * 70)
    log.info(
        "%-20s  %8s  %8s  %8s  %8s", "metric", "mean", "std", "min", "max",
    )
    for key in metric_keys:
        a = agg[key]
        log.info(
            "%-20s  %8.4f  %8.4f  %8.4f  %8.4f",
            key, a["mean"], a["std"], a["min"], a["max"],
        )

    log.info("")
    log.info("Per-fold mean ± std (test set):")
    log.info(
        "  Valence: r = %.4f ± %.4f, MAE = %.4f ± %.4f",
        agg["valence_r"]["mean"], agg["valence_r"]["std"],
        agg["valence_MAE"]["mean"], agg["valence_MAE"]["std"],
    )
    log.info(
        "  Arousal: r = %.4f ± %.4f, MAE = %.4f ± %.4f",
        agg["arousal_r"]["mean"], agg["arousal_r"]["std"],
        agg["arousal_MAE"]["mean"], agg["arousal_MAE"]["std"],
    )

    coverage_pct = 100.0 * len(pooled_y_arr) / len(texts)
    log.info("")
    log.info(
        "Pooled metrics (predictions from %d tested folds, N=%d, %.0f%% of dataset):",
        num_folds, len(pooled_y_arr), coverage_pct,
    )
    log.info(
        "  Valence: r = %.4f, MAE = %.4f",
        pooled_metrics["valence_r"], pooled_metrics["valence_MAE"],
    )
    log.info(
        "  Arousal: r = %.4f, MAE = %.4f",
        pooled_metrics["arousal_r"], pooled_metrics["arousal_MAE"],
    )
    if num_folds < TENTHS:
        untested_start = num_folds
        untested_end = TENTHS - 1
        log.info(
            "  (note: with num_folds=%d, tenths %d-%d never serve as test, "
            "but still participate in train/dev in every fold — no samples "
            "are unused. Run with --kfold %d if you want every sample "
            "predicted from a held-out model exactly once.)",
            num_folds, untested_start, untested_end, TENTHS,
        )

    summary = {
        "dataset": processor.name,
        "num_folds": num_folds,
        "internal_tenths": TENTHS,
        "ratio": "70/20/10 train/dev/test",
        "region_depth": cfg.region_depth,
        "run_tag": cfg.run_tag,
        "total_samples_tested": int(len(pooled_y_arr)),
        "full_dataset_size": int(len(texts)),
        "aggregated_metrics": agg,
        "pooled_metrics": {k: float(v) for k, v in pooled_metrics.items()},
        "per_fold": per_fold,
    }
    tag_suffix = f"_{cfg.run_tag}" if cfg.run_tag else ""
    summary_path = cfg.logs_dir / (
        f"kfold_{processor.name}_d{cfg.region_depth}{tag_suffix}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("wrote %d-fold CV summary to %s", num_folds, summary_path)

    return summary


# ===========================================================================
# 10b. Paired LSTM vs BiLSTM comparison
# ===========================================================================
def compare_errors(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    label_a: str = "A",
    label_b: str = "B",
) -> Dict[str, float]:
    """Run Mann-Whitney U and Wilcoxon signed-rank on two arrays of
    per-sample absolute errors. Returns a dict of test statistics.

    Mann-Whitney U is the primary test per the experimental protocol;
    Wilcoxon signed-rank is reported as a secondary paired-design
    counterpart since both arrays are evaluated on identical test
    samples in the same order (stronger statistical framing).

    Rank-biserial correlation is reported as the MWU effect size;
    with large N the p-value becomes arbitrarily small for any
    directional difference, so effect size carries the practical
    significance.
    """
    from scipy import stats

    errors_a = np.asarray(errors_a, dtype=np.float64).flatten()
    errors_b = np.asarray(errors_b, dtype=np.float64).flatten()
    if len(errors_a) != len(errors_b):
        raise ValueError(
            f"error arrays must have equal length for paired design; "
            f"got {len(errors_a)} vs {len(errors_b)}"
        )

    # Mann-Whitney U (two-sided). This is the primary test per the
    # experimental protocol.
    mwu = stats.mannwhitneyu(errors_a, errors_b, alternative="two-sided")
    u_stat = float(mwu.statistic)
    mwu_p = float(mwu.pvalue)
    # Rank-biserial correlation effect size:
    # r_rb = 1 − 2U / (n1 · n2)
    # Range [-1, 1]; 0 = no effect; sign indicates which sample has
    # smaller ranks (lower errors).
    n1 = len(errors_a)
    n2 = len(errors_b)
    rank_biserial = 1.0 - (2.0 * u_stat) / (n1 * n2)

    # Wilcoxon signed-rank (paired). Requires discarding exact ties.
    try:
        wilcoxon = stats.wilcoxon(
            errors_a, errors_b, zero_method="wilcox",
            alternative="two-sided",
        )
        w_stat = float(wilcoxon.statistic)
        w_p = float(wilcoxon.pvalue)
    except ValueError as exc:
        log.warning("wilcoxon could not be computed: %s", exc)
        w_stat = float("nan")
        w_p = float("nan")

    mean_a = float(errors_a.mean())
    mean_b = float(errors_b.mean())
    median_a = float(np.median(errors_a))
    median_b = float(np.median(errors_b))

    result = {
        "n": int(n1),
        f"mean_error_{label_a}": mean_a,
        f"mean_error_{label_b}": mean_b,
        f"median_error_{label_a}": median_a,
        f"median_error_{label_b}": median_b,
        "mannwhitney_u": u_stat,
        "mannwhitney_p": mwu_p,
        "rank_biserial_r": float(rank_biserial),
        "wilcoxon_w": w_stat,
        "wilcoxon_p": w_p,
    }
    return result


def _set_single_dim_depth(cfg: Config, dataset_name: str, dimension: str) -> None:
    """Resolve the optimal region depth for (dataset, dimension) from
    the registry and set it on cfg along with the target dimension.
    If the caller already set cfg.region_depth explicitly (non-default),
    respect it — useful when experimenting on datasets not in the
    registry.
    """
    cfg.target_dimension = dimension
    if dataset_name in DATASET_OPTIMAL_DEPTH and dimension in DATASET_OPTIMAL_DEPTH[dataset_name]:
        depth = DATASET_OPTIMAL_DEPTH[dataset_name][dimension]
        cfg.region_depth = depth
        log.info(
            "registry: %s/%s → depth=%d", dataset_name, dimension, depth,
        )
    else:
        log.info(
            "registry: no entry for %s/%s; using cfg.region_depth=%d",
            dataset_name, dimension, cfg.region_depth,
        )


def run_comparison(
    processor: BaseDatasetProcessor,
    cfg: Config,
    dimension: str,
    num_folds: int = 5,
) -> Dict[str, Any]:
    """Paired LSTM vs BiLSTM comparison for a single emotion dimension.

    For each of ``num_folds`` folds the pipeline runs preprocessing once
    (parse, regions, vocabulary, embedding matrix, tensor conversion,
    label normalization — all deterministic per fold), then trains two
    models on byte-identical inputs:
        * model A: unidirectional LSTM (paper baseline)
        * model B: bidirectional LSTM (proposed)
    Both models use the same per-fold seed for weight init so that the
    only controlled difference is the LSTM direction.

    Per-fold predictions are saved as a single .npz file per fold
    containing y_true, y_pred_lstm, y_pred_bilstm, and the test indices.
    After all folds are complete, per-sample absolute errors are pooled
    across folds (n≈5120 per model) and fed to
    ``compare_errors`` for Mann-Whitney U and Wilcoxon signed-rank tests.
    """
    import copy as _copy

    TENTHS = 10

    if dimension not in ("valence", "arousal"):
        raise ValueError(f"dimension must be 'valence' or 'arousal'; got {dimension!r}")
    if num_folds < 2 or num_folds > TENTHS:
        raise ValueError(f"num_folds must be in [2, {TENTHS}]; got {num_folds}.")

    # Ensure output directories exist.
    for d in (cfg.cache_dir, cfg.model_dir, cfg.logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    predictions_dir = cfg.logs_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Configure the base cfg with the dimension-specific depth.
    _set_single_dim_depth(cfg, processor.name, dimension)

    log.info("=" * 70)
    log.info(
        "=== LSTM vs BiLSTM paired comparison: dataset=%s, dimension=%s, "
        "depth=%d, num_folds=%d (70/20/10 per fold) ===",
        processor.name, dimension, cfg.region_depth, num_folds,
    )
    log.info("=" * 70)

    # One-shot preprocessing shared by all folds.
    df = processor.load()
    texts = processor.get_texts(df)
    y = processor.get_labels(df)

    parser = TreeParser(cfg.cache_dir, processor)
    trees = parser.parse_all(texts)

    log.info("extracting regions at depth=%d ...", cfg.region_depth)
    all_regions = [text_trees_to_regions(t, cfg.region_depth) for t in trees]

    region_counts = np.array([len(r) for r in all_regions])
    region_lens = np.array([len(r) for text in all_regions for r in text])
    log.info(
        "regions/text: mean=%.2f, max=%d | tokens/region: mean=%.2f, max=%d",
        region_counts.mean(), region_counts.max(),
        region_lens.mean() if len(region_lens) else 0.0,
        region_lens.max() if len(region_lens) else 0,
    )
    maybe_auto_shape(all_regions, cfg)

    # Partition once.
    tenths = _make_kfolds(len(texts), TENTHS, cfg.random_seed)

    per_fold_lstm: List[Dict[str, Any]] = []
    per_fold_bilstm: List[Dict[str, Any]] = []
    pooled_y: List[np.ndarray] = []
    pooled_pred_lstm: List[np.ndarray] = []
    pooled_pred_bilstm: List[np.ndarray] = []

    for k in range(num_folds):
        log.info("")
        log.info("=" * 70)
        log.info("--- fold %d/%d (paired LSTM vs BiLSTM) ---", k + 1, num_folds)
        log.info("=" * 70)

        test_idx = tenths[k]
        dev_a = (k + 1) % TENTHS
        dev_b = (k + 2) % TENTHS
        dev_idx = np.concatenate([tenths[dev_a], tenths[dev_b]])
        train_tenths = [
            tenths[i] for i in range(TENTHS)
            if i not in (k, dev_a, dev_b)
        ]
        train_idx = np.concatenate(train_tenths)

        log.info(
            "splits: train=%d (%.1f%%), dev=%d (%.1f%%), test=%d (%.1f%%)",
            len(train_idx), 100.0 * len(train_idx) / len(texts),
            len(dev_idx), 100.0 * len(dev_idx) / len(texts),
            len(test_idx), 100.0 * len(test_idx) / len(texts),
        )
        log.info(
            "  test=tenth %d, dev=tenths %d+%d, train=remaining 7 tenths",
            k, dev_a, dev_b,
        )

        # --- train LSTM (baseline) -----------------------------------
        log.info("")
        log.info("--- fold %d: training LSTM (baseline) ---", k + 1)
        lstm_cfg = _copy.deepcopy(cfg)
        lstm_cfg.bidirectional_lstm = False
        # Same data-fold seed for both models: only architecture differs.
        lstm_cfg.random_seed = cfg.random_seed + k
        lstm_report, y_test_arr, y_pred_lstm = _run_core(
            processor, lstm_cfg, texts, y, all_regions,
            train_idx, dev_idx, test_idx,
            write_metrics_file=True,
            fold_tag=f"fold{k + 1}",
            model_tag="lstm",
        )
        lstm_report["fold"] = k + 1

        # --- train BiLSTM (proposed) --------------------------------
        log.info("")
        log.info("--- fold %d: training BiLSTM (proposed) ---", k + 1)
        bilstm_cfg = _copy.deepcopy(cfg)
        bilstm_cfg.bidirectional_lstm = True
        bilstm_cfg.random_seed = cfg.random_seed + k
        bilstm_report, y_test_arr_bi, y_pred_bilstm = _run_core(
            processor, bilstm_cfg, texts, y, all_regions,
            train_idx, dev_idx, test_idx,
            write_metrics_file=True,
            fold_tag=f"fold{k + 1}",
            model_tag="bilstm",
        )
        bilstm_report["fold"] = k + 1

        # Sanity: the two models must have been evaluated on identical
        # y_test arrays.
        assert np.array_equal(y_test_arr, y_test_arr_bi), (
            "test labels diverged between LSTM and BiLSTM runs"
        )

        # --- save paired .npz ---------------------------------------
        tag_suffix = f"_{cfg.run_tag}" if cfg.run_tag else ""
        npz_path = (
            predictions_dir
            / f"predictions_{processor.name}_{dimension}_compare_fold{k + 1}{tag_suffix}.npz"
        )
        np.savez(
            npz_path,
            y_true=y_test_arr.flatten(),
            y_pred_lstm=y_pred_lstm.flatten(),
            y_pred_bilstm=y_pred_bilstm.flatten(),
            test_indices=test_idx,
        )
        log.info("wrote paired predictions to %s", npz_path)

        per_fold_lstm.append(lstm_report)
        per_fold_bilstm.append(bilstm_report)
        pooled_y.append(y_test_arr.flatten())
        pooled_pred_lstm.append(y_pred_lstm.flatten())
        pooled_pred_bilstm.append(y_pred_bilstm.flatten())

    # --- select & save the best-across-folds model per architecture ----
    # Selection criterion: highest dev-set Pearson r on the target
    # dimension. We intentionally use DEV r (not test r): dev is the
    # held-out signal each model already optimized against via early
    # stopping, so selecting on it does not introduce any new test
    # leakage. dev MAE is recorded as a tiebreaker but not used in
    # selection because r is the metric the project reports.
    import shutil

    def _pick_best(
        reports: List[Dict[str, Any]], model_tag: str
    ) -> Dict[str, Any]:
        dev_r_key = f"dev_{dimension}_r"
        best_idx = int(
            np.argmax([r[dev_r_key] for r in reports])
        )
        best_report = reports[best_idx]
        best_fold = best_report["fold"]

        src = processor.fold_checkpoint_path(
            model_dir=cfg.model_dir,
            dimension=dimension,
            model_tag=model_tag,
            fold=best_fold,
            region_depth=cfg.region_depth,
            run_tag=cfg.run_tag,
        )
        dst = processor.best_model_path(
            model_dir=cfg.model_dir,
            dimension=dimension,
            model_tag=model_tag,
            region_depth=cfg.region_depth,
            run_tag=cfg.run_tag,
        )
        if src.exists():
            shutil.copy2(src, dst)
            log.info(
                "saved best %s model: fold %d (dev_%s_r=%.4f) → %s",
                model_tag, best_fold, dimension,
                best_report[dev_r_key], dst,
            )
        else:
            log.warning(
                "best %s fold checkpoint missing at %s; skipping copy",
                model_tag, src,
            )

        return {
            "selected_fold": int(best_fold),
            "selection_metric": f"dev_{dimension}_r",
            "selection_value": float(best_report[dev_r_key]),
            "selection_dev_mae": float(best_report[f"dev_{dimension}_MAE"]),
            "selection_test_r": float(best_report[f"{dimension}_r"]),
            "selection_test_mae": float(best_report[f"{dimension}_MAE"]),
            "checkpoint_path": str(dst),
        }

    log.info("")
    log.info(
        "--- selecting best-across-folds model per architecture ---"
    )
    best_lstm_info = _pick_best(per_fold_lstm, "lstm")
    best_bilstm_info = _pick_best(per_fold_bilstm, "bilstm")

    # --- aggregate per-fold metrics ----------------------------------
    def _agg(reports: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in keys:
            vals = np.array([r[key] for r in reports], dtype=np.float64)
            out[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "values": [float(v) for v in vals],
            }
        return out

    metric_keys = [
        f"{dimension}_r", f"{dimension}_MAE",
        f"dev_{dimension}_r", f"dev_{dimension}_MAE",
        "final_val_loss",
    ]
    agg_lstm = _agg(per_fold_lstm, metric_keys)
    agg_bilstm = _agg(per_fold_bilstm, metric_keys)

    # --- pooled metrics + statistical comparison ----------------------
    pooled_y_arr = np.concatenate(pooled_y, axis=0)
    pooled_lstm_arr = np.concatenate(pooled_pred_lstm, axis=0)
    pooled_bilstm_arr = np.concatenate(pooled_pred_bilstm, axis=0)

    pooled_lstm_metrics = evaluate_single_dim(
        pooled_y_arr, pooled_lstm_arr, dimension
    )
    pooled_bilstm_metrics = evaluate_single_dim(
        pooled_y_arr, pooled_bilstm_arr, dimension
    )

    abs_err_lstm = np.abs(pooled_y_arr - pooled_lstm_arr)
    abs_err_bilstm = np.abs(pooled_y_arr - pooled_bilstm_arr)
    stat_result = compare_errors(
        abs_err_lstm, abs_err_bilstm, label_a="lstm", label_b="bilstm"
    )

    # --- report -------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info(
        "=== LSTM vs BiLSTM summary: %s / %s / depth=%d / %d folds ===",
        processor.name, dimension, cfg.region_depth, num_folds,
    )
    log.info("=" * 70)

    log.info("Per-fold mean ± std (test set):")
    log.info(
        "  LSTM   %s: r = %.4f ± %.4f, MAE = %.4f ± %.4f",
        dimension,
        agg_lstm[f"{dimension}_r"]["mean"], agg_lstm[f"{dimension}_r"]["std"],
        agg_lstm[f"{dimension}_MAE"]["mean"], agg_lstm[f"{dimension}_MAE"]["std"],
    )
    log.info(
        "  BiLSTM %s: r = %.4f ± %.4f, MAE = %.4f ± %.4f",
        dimension,
        agg_bilstm[f"{dimension}_r"]["mean"], agg_bilstm[f"{dimension}_r"]["std"],
        agg_bilstm[f"{dimension}_MAE"]["mean"], agg_bilstm[f"{dimension}_MAE"]["std"],
    )

    coverage_pct = 100.0 * len(pooled_y_arr) / len(texts)
    log.info("")
    log.info(
        "Pooled across %d folds (N=%d, %.0f%% of dataset):",
        num_folds, len(pooled_y_arr), coverage_pct,
    )
    log.info(
        "  LSTM   %s: r = %.4f, MAE = %.4f",
        dimension, pooled_lstm_metrics[f"{dimension}_r"],
        pooled_lstm_metrics[f"{dimension}_MAE"],
    )
    log.info(
        "  BiLSTM %s: r = %.4f, MAE = %.4f",
        dimension, pooled_bilstm_metrics[f"{dimension}_r"],
        pooled_bilstm_metrics[f"{dimension}_MAE"],
    )

    log.info("")
    log.info("Statistical comparison on per-sample absolute errors (n=%d each):",
             stat_result["n"])
    log.info(
        "  LSTM   mean |err| = %.4f, median = %.4f",
        stat_result["mean_error_lstm"], stat_result["median_error_lstm"],
    )
    log.info(
        "  BiLSTM mean |err| = %.4f, median = %.4f",
        stat_result["mean_error_bilstm"], stat_result["median_error_bilstm"],
    )
    log.info(
        "  Mann-Whitney U = %.1f, p = %.4g, rank-biserial r = %.4f",
        stat_result["mannwhitney_u"], stat_result["mannwhitney_p"],
        stat_result["rank_biserial_r"],
    )
    log.info(
        "  Wilcoxon signed-rank W = %.1f, p = %.4g (paired, same test samples)",
        stat_result["wilcoxon_w"], stat_result["wilcoxon_p"],
    )
    mwu_sig = stat_result["mannwhitney_p"] < 0.05
    log.info(
        "  → MWU %s significant at α=0.05",
        "IS" if mwu_sig else "is NOT",
    )

    summary = {
        "dataset": processor.name,
        "dimension": dimension,
        "region_depth": cfg.region_depth,
        "num_folds": num_folds,
        "internal_tenths": TENTHS,
        "ratio": "70/20/10 train/dev/test",
        "run_tag": cfg.run_tag,
        "total_samples_tested": int(len(pooled_y_arr)),
        "full_dataset_size": int(len(texts)),
        "lstm": {
            "per_fold_metrics": agg_lstm,
            "pooled_metrics": {k: float(v) for k, v in pooled_lstm_metrics.items()},
            "best_of_folds": best_lstm_info,
            "per_fold_reports": per_fold_lstm,
        },
        "bilstm": {
            "per_fold_metrics": agg_bilstm,
            "pooled_metrics": {k: float(v) for k, v in pooled_bilstm_metrics.items()},
            "best_of_folds": best_bilstm_info,
            "per_fold_reports": per_fold_bilstm,
        },
        "statistical_test": stat_result,
        "significant_at_0.05_mwu": bool(mwu_sig),
    }
    tag_suffix = f"_{cfg.run_tag}" if cfg.run_tag else ""
    summary_path = (
        cfg.logs_dir
        / f"compare_{processor.name}_{dimension}_d{cfg.region_depth}{tag_suffix}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("wrote comparison summary to %s", summary_path)

    return summary


def run_comparison_predefined_split(
    processor: BaseDatasetProcessor,
    cfg: Config,
    dimension: str,
) -> Dict[str, Any]:
    """Paired LSTM vs BiLSTM comparison for a dataset that provides its
    own canonical train/dev/test split (SST).

    Difference from ``run_comparison``: a single training run per
    architecture instead of a 5-fold loop. Dataset's
    ``predefined_split(df)`` returns (train_idx, dev_idx, test_idx);
    both models train on byte-identical inputs (same data, same
    preprocessing, same seed) so the only controlled difference is the
    LSTM direction.

    Per-sample absolute errors on the test set are fed to
    ``compare_errors`` for Mann-Whitney U and Wilcoxon signed-rank.
    The "best-of-folds" concept collapses to a single stored model per
    architecture; ``best_model_path`` is still used so the artifact
    layout matches other datasets.
    """
    import copy as _copy
    import shutil

    # Ensure output directories exist.
    for d in (cfg.cache_dir, cfg.model_dir, cfg.logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    predictions_dir = cfg.logs_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Configure the base cfg with the dimension-specific depth.
    _set_single_dim_depth(cfg, processor.name, dimension)

    # SST is valence-only. Guard arousal requests.
    if dimension == "arousal" and processor.name == "sst":
        raise SystemExit(
            f"[{processor.name}] has no arousal labels; use --dimension "
            f"valence."
        )

    log.info("=" * 70)
    log.info(
        "=== LSTM vs BiLSTM comparison (predefined split): dataset=%s, "
        "dimension=%s, depth=%d ===",
        processor.name, dimension, cfg.region_depth,
    )
    log.info("=" * 70)

    # Load + parse + regions once (same preprocessing for both models).
    df = processor.load()
    texts = processor.get_texts(df)
    y = processor.get_labels(df)

    # Resolve canonical split.
    split = processor.predefined_split(df)
    if split is None:
        raise SystemExit(
            f"[{processor.name}] has no predefined split; use "
            f"run_comparison (5-fold CV) instead."
        )
    train_idx, dev_idx, test_idx = split
    log.info(
        "splits (dataset-provided): train=%d (%.1f%%), dev=%d (%.1f%%), "
        "test=%d (%.1f%%)",
        len(train_idx), 100.0 * len(train_idx) / len(df),
        len(dev_idx),   100.0 * len(dev_idx) / len(df),
        len(test_idx),  100.0 * len(test_idx) / len(df),
    )

    parser = TreeParser(cfg.cache_dir, processor)
    trees = parser.parse_all(texts)

    log.info("extracting regions at depth=%d ...", cfg.region_depth)
    all_regions = [text_trees_to_regions(t, cfg.region_depth) for t in trees]

    region_counts = np.array([len(r) for r in all_regions])
    region_lens = np.array([len(r) for text in all_regions for r in text])
    log.info(
        "regions/text: mean=%.2f, max=%d | tokens/region: mean=%.2f, max=%d",
        region_counts.mean(), region_counts.max(),
        region_lens.mean() if len(region_lens) else 0.0,
        region_lens.max() if len(region_lens) else 0,
    )
    maybe_auto_shape(all_regions, cfg)

    # --- train LSTM (baseline) ---------------------------------------
    log.info("")
    log.info("--- training LSTM (baseline) ---")
    lstm_cfg = _copy.deepcopy(cfg)
    lstm_cfg.bidirectional_lstm = False
    # Use same seed for both models so weight init differs only in
    # direction of the recurrent layer.
    lstm_cfg.random_seed = cfg.random_seed
    lstm_report, y_test_arr, y_pred_lstm = _run_core(
        processor, lstm_cfg, texts, y, all_regions,
        train_idx, dev_idx, test_idx,
        write_metrics_file=True,
        fold_tag=None,  # no fold concept on predefined splits
        model_tag="lstm",
    )

    # --- train BiLSTM (proposed) -------------------------------------
    log.info("")
    log.info("--- training BiLSTM (proposed) ---")
    bilstm_cfg = _copy.deepcopy(cfg)
    bilstm_cfg.bidirectional_lstm = True
    bilstm_cfg.random_seed = cfg.random_seed
    bilstm_report, y_test_arr_bi, y_pred_bilstm = _run_core(
        processor, bilstm_cfg, texts, y, all_regions,
        train_idx, dev_idx, test_idx,
        write_metrics_file=True,
        fold_tag=None,
        model_tag="bilstm",
    )

    # Sanity: same test labels in both runs.
    assert np.array_equal(y_test_arr, y_test_arr_bi), (
        "test labels diverged between LSTM and BiLSTM runs"
    )

    # --- save paired .npz --------------------------------------------
    tag_suffix = f"_{cfg.run_tag}" if cfg.run_tag else ""
    npz_path = (
        predictions_dir
        / f"predictions_{processor.name}_{dimension}_compare{tag_suffix}.npz"
    )
    np.savez(
        npz_path,
        y_true=y_test_arr.flatten(),
        y_pred_lstm=y_pred_lstm.flatten(),
        y_pred_bilstm=y_pred_bilstm.flatten(),
        test_indices=test_idx,
    )
    log.info("wrote paired predictions to %s", npz_path)

    # --- copy each checkpoint to the canonical best_model_path -------
    # For consistency with other datasets, we use the same
    # best_model_path convention. On a predefined split there's only
    # one trained model per architecture, so "best of folds" is
    # trivially that single model.
    def _save_best_model(model_tag: str, report: Dict[str, Any]) -> Dict[str, Any]:
        src = processor.fold_checkpoint_path(
            model_dir=cfg.model_dir,
            dimension=dimension,
            model_tag=model_tag,
            fold=None,
            region_depth=cfg.region_depth,
            run_tag=cfg.run_tag,
        )
        dst = processor.best_model_path(
            model_dir=cfg.model_dir,
            dimension=dimension,
            model_tag=model_tag,
            region_depth=cfg.region_depth,
            run_tag=cfg.run_tag,
        )
        if src.exists():
            shutil.copy2(src, dst)
            log.info(
                "saved best %s model: dev_%s_r=%.4f → %s",
                model_tag, dimension,
                report[f"dev_{dimension}_r"], dst,
            )
        else:
            log.warning(
                "best %s checkpoint missing at %s; skipping copy",
                model_tag, src,
            )
        return {
            "selection_metric": f"dev_{dimension}_r",
            "selection_value": float(report[f"dev_{dimension}_r"]),
            "selection_dev_mae": float(report[f"dev_{dimension}_MAE"]),
            "selection_test_r": float(report[f"{dimension}_r"]),
            "selection_test_mae": float(report[f"{dimension}_MAE"]),
            "checkpoint_path": str(dst),
        }

    log.info("")
    log.info("--- saving models under canonical best_model_path ---")
    best_lstm_info = _save_best_model("lstm", lstm_report)
    best_bilstm_info = _save_best_model("bilstm", bilstm_report)

    # --- statistical comparison --------------------------------------
    y_true_flat = y_test_arr.flatten()
    pred_lstm_flat = y_pred_lstm.flatten()
    pred_bilstm_flat = y_pred_bilstm.flatten()

    test_metrics_lstm = evaluate_single_dim(y_true_flat, pred_lstm_flat, dimension)
    test_metrics_bilstm = evaluate_single_dim(y_true_flat, pred_bilstm_flat, dimension)

    abs_err_lstm = np.abs(y_true_flat - pred_lstm_flat)
    abs_err_bilstm = np.abs(y_true_flat - pred_bilstm_flat)
    stat_result = compare_errors(
        abs_err_lstm, abs_err_bilstm, label_a="lstm", label_b="bilstm"
    )

    # --- report -------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info(
        "=== LSTM vs BiLSTM summary: %s / %s / depth=%d / predefined split ===",
        processor.name, dimension, cfg.region_depth,
    )
    log.info("=" * 70)

    log.info("Test set performance (N=%d):", len(y_true_flat))
    log.info(
        "  LSTM   %s: r = %.4f, MAE = %.4f",
        dimension,
        test_metrics_lstm[f"{dimension}_r"],
        test_metrics_lstm[f"{dimension}_MAE"],
    )
    log.info(
        "  BiLSTM %s: r = %.4f, MAE = %.4f",
        dimension,
        test_metrics_bilstm[f"{dimension}_r"],
        test_metrics_bilstm[f"{dimension}_MAE"],
    )

    log.info("")
    log.info(
        "Statistical comparison on per-sample absolute errors (n=%d each):",
        stat_result["n"],
    )
    log.info(
        "  LSTM   mean |err| = %.4f, median = %.4f",
        stat_result["mean_error_lstm"], stat_result["median_error_lstm"],
    )
    log.info(
        "  BiLSTM mean |err| = %.4f, median = %.4f",
        stat_result["mean_error_bilstm"], stat_result["median_error_bilstm"],
    )
    log.info(
        "  Mann-Whitney U = %.1f, p = %.4g, rank-biserial r = %.4f",
        stat_result["mannwhitney_u"], stat_result["mannwhitney_p"],
        stat_result["rank_biserial_r"],
    )
    log.info(
        "  Wilcoxon signed-rank W = %.1f, p = %.4g (paired, same test samples)",
        stat_result["wilcoxon_w"], stat_result["wilcoxon_p"],
    )
    mwu_sig = stat_result["mannwhitney_p"] < 0.05
    log.info(
        "  → MWU %s significant at α=0.05",
        "IS" if mwu_sig else "is NOT",
    )

    summary = {
        "dataset": processor.name,
        "dimension": dimension,
        "region_depth": cfg.region_depth,
        "evaluation_protocol": "predefined_split",
        "split_source": "dataset-provided 'split' column",
        "run_tag": cfg.run_tag,
        "train_size": int(len(train_idx)),
        "dev_size": int(len(dev_idx)),
        "test_size": int(len(test_idx)),
        "full_dataset_size": int(len(df)),
        "lstm": {
            "test_metrics": {k: float(v) for k, v in test_metrics_lstm.items()},
            "report": lstm_report,
            "best_model": best_lstm_info,
        },
        "bilstm": {
            "test_metrics": {k: float(v) for k, v in test_metrics_bilstm.items()},
            "report": bilstm_report,
            "best_model": best_bilstm_info,
        },
        "statistical_test": stat_result,
        "significant_at_0.05_mwu": bool(mwu_sig),
    }
    summary_path = (
        cfg.logs_dir
        / f"compare_{processor.name}_{dimension}_d{cfg.region_depth}{tag_suffix}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("wrote comparison summary to %s", summary_path)

    return summary


# ===========================================================================
# 11. CLI entry point
# ===========================================================================
def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["emobank_writer", "emobank_reader", "fb", "cvat", "sst"],
        default="emobank_writer",
        help="which dataset processor to run.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="path to dataset CSV (defaults to BASE_DIR/<default_name>.csv).",
    )
    parser.add_argument(
        "--embedding-path",
        default=None,
        help="override path to the pre-trained word embedding file. When "
             "set, replaces the processor's default embedding file "
             "(English GloVe for EmoBank/FB, Chinese-Word-Vectors for "
             "CVAT). The file must be a whitespace-separated text file "
             "of '<token> <v1> ... <v300>' lines, with an optional "
             "word2vec-style '<vocab_size> <dim>' header (GloVe and "
             "FastText .vec formats are both supported).",
    )
    parser.add_argument("--region-depth", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--embedding-dropout", type=float, default=None)
    parser.add_argument("--spatial-dropout", type=float, default=None)
    parser.add_argument("--recurrent-dropout", type=float, default=None)
    parser.add_argument("--post-lstm-dropout", type=float, default=None)
    parser.add_argument("--l2-reg", type=float, default=None)
    parser.add_argument(
        "--loss",
        choices=["mse", "ccc"],
        default=None,
        help="training loss: 'mse' (paper) or 'ccc' (Concordance Correlation Coefficient).",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="use a bidirectional LSTM for the sequential layer.",
    )
    parser.add_argument(
        "--multi-kernel",
        action="store_true",
        help="use parallel multi-kernel CNN branches (3,4,5) × 100 filters "
             "instead of the paper's single kernel (3) × 60 filters. The "
             "authors' published code uses this even though the paper "
             "describes a single kernel.",
    )
    parser.add_argument(
        "--trainable-embeddings",
        action="store_true",
        help="fine-tune GloVe embeddings during training (adds ~4M trainable params).",
    )
    parser.add_argument(
        "--no-normalize-labels",
        action="store_true",
        help="disable V/A standardization during training.",
    )
    parser.add_argument(
        "--depth-sweep",
        type=str,
        default=None,
        help="comma-separated list of depths to sweep, e.g. '1,2,3,4,5,6'; "
             "trains at each, picks best by dev loss, reports test.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="comma-separated list of seeds for SEEDED REPEATS (NOT CV): "
             "each seed gets its own random 70/20/10 split, which means "
             "test sets can overlap across seeds. For real non-overlapping "
             "cross-validation, use --kfold instead.",
    )
    parser.add_argument(
        "--kfold",
        type=int,
        default=None,
        help="run REAL N-fold cross-validation: partitions the dataset "
             "into N non-overlapping folds, tests on each fold exactly "
             "once, reports per-fold mean ± std AND pooled metrics over "
             "all N predictions. Paper uses 5.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="run paired LSTM vs BiLSTM comparison for a single emotion "
             "dimension (--dimension required). For each of 5 folds "
             "trains both models on identical splits/vocab/embeddings, "
             "saves per-sample predictions as .npz files, and reports "
             "Mann-Whitney U + Wilcoxon signed-rank tests on pooled "
             "per-sample absolute errors (n≈5120 per model).",
    )
    parser.add_argument(
        "--dimension",
        choices=["valence", "arousal"],
        default=None,
        help="target emotion dimension. When set (required for --compare, "
             "optional otherwise), the pipeline trains a single-output "
             "regressor and looks up the dataset's optimal region depth "
             "from DATASET_OPTIMAL_DEPTH (emobank_writer: V=4, A=5). "
             "--region-depth overrides the registry lookup.",
    )
    parser.add_argument(
        "--compare-folds",
        type=int,
        default=5,
        help="number of folds for --compare (default 5, matches paper). "
             "Internal partition is always 10 tenths; this controls how "
             "many of the 10 test-tenths to cycle through.",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="preserve word casing in vocabulary and GloVe lookup (GloVe 840B is cased).",
    )
    parser.add_argument(
        "--paper-config",
        action="store_true",
        help="reset regularization to the paper's original values for a direct replication run.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="optional run identifier appended to all output filenames "
             "and model subfolders. Use this to keep runs with different "
             "configurations on disk simultaneously — e.g. '--run-tag "
             "tuned' for one run and '--paper-config --run-tag paper' "
             "for another. When omitted, file paths match legacy layout "
             "exactly (no tag folder, no filename suffix).",
    )
    args = parser.parse_args(argv)

    cfg = Config()
    if args.region_depth is not None:
        cfg.region_depth = args.region_depth
    if args.embedding_path is not None:
        cfg.embedding_path_override = Path(args.embedding_path)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.seed is not None:
        cfg.random_seed = args.seed
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.embedding_dropout is not None:
        cfg.embedding_dropout = args.embedding_dropout
    if args.spatial_dropout is not None:
        cfg.spatial_dropout = args.spatial_dropout
    if args.recurrent_dropout is not None:
        cfg.recurrent_dropout = args.recurrent_dropout
    if args.post_lstm_dropout is not None:
        cfg.post_lstm_dropout = args.post_lstm_dropout
    if args.l2_reg is not None:
        cfg.l2_reg = args.l2_reg
    if args.loss is not None:
        cfg.loss_type = args.loss
    if args.bidirectional:
        cfg.bidirectional_lstm = True
    if args.multi_kernel:
        cfg.multi_kernel = True
    if args.trainable_embeddings:
        cfg.trainable_embeddings = True
    if args.no_normalize_labels:
        cfg.normalize_labels = False
    if args.case_sensitive:
        cfg.case_sensitive = True
    if args.paper_config:
        cfg.embedding_dropout = 0.0
        cfg.spatial_dropout = 0.25
        cfg.recurrent_dropout = 0.25
        cfg.post_lstm_dropout = 0.0
        cfg.l2_reg = 0.0
        cfg.grad_clip_norm = 0.0
        cfg.learning_rate = 1e-3
        cfg.epochs = 20
        cfg.early_stopping_patience = 5
        cfg.normalize_labels = False
        cfg.trainable_embeddings = False
        cfg.bidirectional_lstm = False
        cfg.loss_type = "mse"
        cfg.case_sensitive = False
        cfg.multi_kernel = False
        cfg.target_dimension = None

    # Apply run tag AFTER --paper-config so the tag scopes the run
    # regardless of which preset was chosen.
    if args.run_tag:
        cfg.run_tag = args.run_tag
        log.info(
            "run_tag='%s' — all outputs will be scoped under this tag",
            cfg.run_tag,
        )

    # Dataset dispatch. Each entry maps a --dataset choice to its
    # processor class. The default CSV filename lives on the class
    # (default_filename attribute), so adding a new dataset is a
    # one-line addition here plus a one-class definition above plus a
    # DATASET_OPTIMAL_DEPTH entry.
    _DATASET_REGISTRY = {
        "emobank_writer": EmoBankWriterProcessor,
        "emobank_reader": EmoBankReaderProcessor,
        "fb": FBProcessor,
        "cvat": CVATProcessor,
        "sst": SSTProcessor,
    }
    if args.dataset in _DATASET_REGISTRY:
        proc_cls = _DATASET_REGISTRY[args.dataset]
        csv_path = (
            Path(args.csv) if args.csv
            else cfg.base_dir / proc_cls.default_filename
        )
        processor: BaseDatasetProcessor = proc_cls(csv_path)
    else:
        raise SystemExit(f"unsupported dataset: {args.dataset}")

    # Apply --dimension to cfg. When used outside --compare, this turns
    # on single-dim training with the dataset's optimal depth looked up
    # from the registry (unless --region-depth was explicitly set, in
    # which case we honor the CLI value).
    if args.dimension and not args.compare:
        explicit_depth = args.region_depth is not None
        _set_single_dim_depth(cfg, processor.name, args.dimension)
        if explicit_depth:
            cfg.region_depth = args.region_depth
            log.info(
                "--region-depth=%d overrides registry for %s/%s",
                args.region_depth, processor.name, args.dimension,
            )

    # Mutual-exclusion checks.
    mode_flags = [
        ("--depth-sweep", bool(args.depth_sweep)),
        ("--seeds", bool(args.seeds)),
        ("--kfold", args.kfold is not None),
        ("--compare", bool(args.compare)),
    ]
    active = [name for name, v in mode_flags if v]
    if len(active) > 1:
        raise SystemExit(
            f"mutually exclusive flags given: {active}. "
            "Pick one mode: --depth-sweep (find best depth), "
            "--kfold / --seeds (CV averaging at a chosen depth), "
            "--compare (paired LSTM vs BiLSTM), "
            "or none (single split)."
        )

    if args.compare:
        if not args.dimension:
            raise SystemExit(
                "--compare requires --dimension {valence|arousal} to "
                "select the emotion dimension (and its optimal depth)."
            )
        if args.compare_folds < 2:
            raise SystemExit("--compare-folds must be >= 2")
        # Reject --dimension that the dataset doesn't support. This
        # catches e.g. --dataset sst --dimension arousal before any
        # training time is wasted.
        dataset_dims = DATASET_OPTIMAL_DEPTH.get(args.dataset, {})
        if args.dimension not in dataset_dims:
            raise SystemExit(
                f"[{args.dataset}] does not have '{args.dimension}' "
                f"labels; supported dimensions for this dataset: "
                f"{sorted(dataset_dims.keys())}."
            )
        # If the user explicitly set --region-depth, honor it over the
        # registry lookup inside the comparison function.
        if args.region_depth is not None:
            cfg.region_depth = args.region_depth
            log.info(
                "--region-depth=%d will override registry in comparison",
                args.region_depth,
            )
        # Decide which comparison path to run based on whether the
        # processor has a canonical split. SST returns a non-None
        # predefined_split and skips k-fold; all other datasets fall
        # through to the paper's 5-fold CV protocol. We check the
        # method override via MRO so we don't have to load the CSV
        # just to find out.
        processor_has_predefined_split = (
            type(processor).predefined_split
            is not BaseDatasetProcessor.predefined_split
        )
        if processor_has_predefined_split:
            log.info(
                "dataset '%s' provides a canonical train/dev/test split; "
                "dispatching to run_comparison_predefined_split "
                "(no k-fold CV)",
                processor.name,
            )
            run_comparison_predefined_split(
                processor, cfg,
                dimension=args.dimension,
            )
        else:
            run_comparison(
                processor, cfg,
                dimension=args.dimension,
                num_folds=args.compare_folds,
            )
    elif args.depth_sweep:
        depths = [int(x.strip()) for x in args.depth_sweep.split(",") if x.strip()]
        run_depth_sweep(processor, cfg, depths)
    elif args.kfold is not None:
        if args.kfold < 2:
            raise SystemExit("--kfold must be >= 2")
        run_kfold(processor, cfg, num_folds=args.kfold)
    elif args.seeds:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        run_multi_seed(processor, cfg, seeds)
    else:
        run(processor, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())