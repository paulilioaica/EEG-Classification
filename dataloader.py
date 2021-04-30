import os
import torch
from torch_geometric.data import Dataset
import numpy as np
from torch_geometric.data import Data

LABELS = ['Valence', 'Arousal', 'Dominance']


class DEAP(Dataset):
    def __init__(self, root_dir, label_path, transform=None, pre_transform=None):
        super(DEAP, self).__init__(None, transform, pre_transform)
        self.label_dir = label_path
        self.root_dir = root_dir
        assert os.path.exists(os.path.join(self.root_dir)), "Path to files cannot be found"
        assert os.path.exists(os.path.join(self.label_dir)), "Path to labels cannot be found"
        self.paths = sorted([os.path.join(self.root_dir, path) for path in os.listdir(self.root_dir) if 'data' in path])
        self.labels = sorted(
            [os.path.join(self.label_dir, path) for path in os.listdir(self.root_dir) if 'label' in path])

    def len(self):
        return len(self.paths)

    def get(self, item):
        eeg_data = np.load(self.paths[item])
        label = np.load(self.labels[item])

        corr_matrix = np.corrcoef(eeg_data[:32, :])
        corr_matrix = np.abs(corr_matrix)
        adj_matrix = corr_matrix - np.identity(corr_matrix.shape[0])

        unsim_matrix = np.zeros((adj_matrix.shape[0], adj_matrix.shape[1]))
        indices = np.triu_indices(adj_matrix.shape[0], k=0)
        unsim_matrix[indices] = adj_matrix[indices]

        corrs = np.where(unsim_matrix >= 0.5)

        edge_index = torch.tensor([[i, j] for i, j in zip(corrs[0], corrs[1])], dtype=torch.long)
        edge_features = torch.tensor([adj_matrix[i][j] for i, j in zip(corrs[0], corrs[1])], dtype=torch.float)

        x = torch.tensor([[eeg_data[i][-1]] if i in set(corrs[0]).intersection(set(corrs[1])) else [0]
                          for i in range(np.max(corrs) + 1)], dtype=torch.float)
        graph = Data(x=x, edge_index=edge_index.t().contiguous(), edge_features=edge_features, y=torch.tensor(label).unsqueeze(0))


        return graph
