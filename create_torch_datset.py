import torch
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, X, U, t, seq_len):
        self.X = X
        self.U = U
        self.t = t
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        
        """batch_idx = np.arange(idx, idx + self.seq_len)
        batch_t = torch.tensor(self.t[batch_idx].astype('float32'))
        batch_u = torch.tensor(self.U[batch_idx].astype('float32'))
        batch_x = torch.tensor(self.X[batch_idx].astype('float32'))
        return batch_t, batch_u, batch_x"""

        
        start_idx = np.random.choice(len(self.X) - self.seq_len)
        idx_range = np.arange(start_idx, start_idx + self.seq_len)

        return torch.tensor(self.t[idx_range], dtype=torch.float32), torch.tensor(self.U[idx_range], dtype=torch.float32), torch.tensor(self.X[idx_range], dtype=torch.float32)
        
  


