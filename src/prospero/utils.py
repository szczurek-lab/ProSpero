import random
import numpy as np
import torch
import hashlib
import os
from prospero.experiments_config import WT_SEQUENCES


def set_seed(seed=0, full_deterministic=False):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if full_deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            torch.use_deterministic_algorithms(True, warn_only=False)
            # Enable CuDNN deterministic mode
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_new_starting_seq(dataset):
    seqs = np.array(dataset.train.tolist() + dataset.valid.tolist())
    scores = np.array(dataset.train_scores.tolist() + dataset.valid_scores.tolist())
    return seqs[np.argmax(scores)]

def get_new_starting_seq_dshift(dataset, task):
    wt_seq = WT_SEQUENCES[task]
    wt_fitness = 0.7045787572860718 if task in ["D_SHIFT", "D_SHIFT_SMALL"] else 0.4911899268627167
    seqs = np.array(dataset.added_sequences + [wt_seq])
    scores =np.array(dataset.added_scores + [wt_fitness])
    return seqs[np.argmax(scores)]

def sample_new_starting_sequences(dataset, n_best_seqs):
    seqs = np.array(dataset.train.tolist() + dataset.valid.tolist())
    scores = np.array(dataset.train_scores.tolist() + dataset.valid_scores.tolist())
    top_indices = np.argsort(scores)[-n_best_seqs:][::-1]
    return seqs[top_indices].tolist()


def calculate_seed(c, nt, s):
    seed_str = f"{c}-{nt}-{s}"
    hash_object = hashlib.sha256(seed_str.encode())
    seed = int(hash_object.hexdigest()[:8], 16)
    return seed


