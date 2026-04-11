"""Data loading for Hillenbrand vowel formant dataset."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).parent.parent / "data"
VOWEL_CSV = DATA_DIR / "hillenbrand_vowels.csv"

ALL_VOWELS = ['a', 'e', 'i', 'o', 'u']


def load_hillenbrand(vowels: list[str] = ['a', 'i'],
                     split: float = 0.8,
                     seed: int = 42,
                     ) -> tuple[list, list, dict]:
    """Load Hillenbrand vowel formant data.

    Args:
        vowels: which vowel classes to include (default: binary a/i).
        split: fraction of data for training.
        seed: random seed for reproducible train/test split.

    Returns:
        (train_data, test_data, info) where each sample is
        (features: np.ndarray, class_idx: int) and info contains
        normalization parameters and class names.
    """
    samples = []
    with open(VOWEL_CSV) as f:
        for row in csv.DictReader(f):
            if row['vowel'] in vowels:
                f1 = float(row['F1'])
                f2 = float(row['F2'])
                cls = vowels.index(row['vowel'])
                samples.append((f1, f2, cls))

    # Split before normalizing to avoid test-set leakage
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    n_train = int(split * len(samples))
    train_raw = [samples[i] for i in idx[:n_train]]
    test_raw = [samples[i] for i in idx[n_train:]]

    # Compute normalization statistics from training set only
    f1_train = np.array([s[0] for s in train_raw])
    f2_train = np.array([s[1] for s in train_raw])
    f1_min, f1_max = f1_train.min(), f1_train.max()
    f2_min, f2_max = f2_train.min(), f2_train.max()

    def normalize(raw_samples):
        out = []
        for f1, f2, cls in raw_samples:
            x = np.array([
                2 * (f1 - f1_min) / (f1_max - f1_min) - 1,
                2 * (f2 - f2_min) / (f2_max - f2_min) - 1,
            ])
            out.append((x, cls))
        return out

    train_data = normalize(train_raw)
    test_data = normalize(test_raw)

    info = {
        "vowels": vowels,
        "n_classes": len(vowels),
        "n_train": len(train_data),
        "n_test": len(test_data),
        "f1_range": (float(f1_min), float(f1_max)),
        "f2_range": (float(f2_min), float(f2_max)),
    }

    return train_data, test_data, info
