import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = len(seqs)
    if N == 0 :
        return np.empty((0,0))

    if max_len is not None:
        L=max_len
    else:
        L=max(len(seq) for seq in seqs)

    result = np.full((N,L),pad_value)

    for i,seq in enumerate(seqs):
        length= min(len(seq),L)
        if length > 0:
            result[i, :length]= seq[:length]


    return result
    pass