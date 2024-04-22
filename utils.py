import torch
import numpy as np



def get_device():
    if torch.cuda.is_available():
      device = "cuda"
    elif torch.backends.mps.is_available():
      device = "mps"
    else:
      device = "cpu"

    return device  


def get_batch(X, U, t, batch_size, seq_len, device):

    # Select batch indexes
    num_train_samples = X.shape[0]
    batch_start = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False) # batch start indices
    batch_idx = batch_start[:, np.newaxis] + np.arange(seq_len) # batch samples indices
    batch_idx = batch_idx.T  # transpose indexes to obtain batches with structure (m, q, n_x)

    # Extract batch data
    batch_t = torch.tensor(t[batch_idx].astype('float32'))
    batch_u = torch.tensor(U[batch_idx].astype('float32'))
    batch_x = torch.tensor(X[batch_idx].astype('float32'))

    return batch_t.to(device), batch_u.to(device), batch_x.to(device)