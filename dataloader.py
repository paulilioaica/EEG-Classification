import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import random

LABELS = ['Valence', 'Arousal', 'Dominance']


class DEAP(Dataset):
    def __init__(self, root_dir, label_path):
        self.label_dir = label_path
        self.root_dir = root_dir
        assert os.path.exists(os.path.join(self.root_dir)), "Path to files cannot be found"
        assert os.path.exists(os.path.join(self.label_dir)), "Path to files cannot be found"
        self.paths = sorted([os.path.join(self.root_dir, path) for path in os.listdir(self.root_dir) if 'data' in path])
        self.labels = sorted([os.path.join(self.label_dir, path) for path in os.listdir(self.root_dir) if 'label' in path])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        eeg_data = np.load(self.path[item])

        corr_matrix = np.corrcoef(eeg_data[:32, :])
        corr_matrix = np.abs(corr_matrix)

        degree_matrix = corr_matrix.sum(axis=1)
        diagonal_matrix = np.diag(degree_matrix)

        L = diagonal_matrix - corr_matrix

        label = np.load(self.labels[item])

        return L, label