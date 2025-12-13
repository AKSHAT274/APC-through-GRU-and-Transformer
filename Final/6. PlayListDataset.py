import torch
from torch.utils.data import Dataset
import pandas as pd


class PlaylistDataset(Dataset):
    '''
    PyTorch Dataset for loading and batching playlist sequences
    '''

    def __init__(self, data, vocab, max_sequence_length=50):
        '''
        Initialize the PlaylistDataset with data and vocabulary
        
        ⭐ Step A — Load the preprocessed data
            ✔ Read Parquet file containing playlist sequences
            ✔ DataFrame has columns: history, target, playlist_id, etc.
            ✔ Store in self.df for efficient indexing
        
        ⭐ Step B — Store hyperparameters
            ✔ vocab_size: collection of all unique track URIs
            ✔ max_sequence_length: maximum history length (default 50)
            ✔ Longer sequences will be truncated
        
        ⭐ Step C — Define special token indices
            ✔ PAD_IDX = 0: for padding shorter sequences
            ✔ UNK_IDX = 1: for unknown/rare tracks not in vocabulary
            ✔ Reserved at start of vocabulary
        
        ⭐ Step D — Build vocabulary mappings
            ✔ track_to_idx: map track URI → integer index (offset by +2)
            ✔ idx_to_track: reverse map for decoding predictions
            ✔ Offset ensures no overlap with special tokens
        '''


    def __len__(self):
        '''
        Return the total number of sequences in the dataset
        
        ✔ Required by PyTorch Dataset interface
        ✔ Returns number of rows in self.df
        ✔ Each row = one training example (history → target)
        '''
    
    def __getitem__(self, idx):
        '''
        Retrieve and encode a single playlist sequence
        
        ⭐ Step A — Load raw sequence from DataFrame
            ✔ Get row at index idx
            ✔ Extract 'history' (list of track URIs) and 'target' (next track URI)
        
        ⭐ Step B — Handle history format
            ✔ If history is string (from Parquet): split by '|' delimiter
            ✔ If history is already a list: use directly
            ✔ Handle empty history gracefully
        
        ⭐ Step C — Convert tracks to vocabulary indices
            ✔ Map each track URI in history to integer index
            ✔ Use track_to_idx.get(track, UNK_IDX) for safe lookup
            ✔ Unknown tracks → UNK_IDX (1)
        
        ⭐ Step D — Convert target to index
            ✔ Map target track URI to integer index
            ✔ Use UNK_IDX if target not in vocabulary
        
        ⭐ Step E — Truncate long sequences
            ✔ If history length > max_sequence_length, keep only last N tokens
            ✔ Preserves most recent context for prediction
            ✔ Prevents memory issues with very long playlists
        
        ⭐ Step F — Return encoded sequence
            ✔ Returns dict with 'history', 'target', 'seq_length'
            ✔ Ready for batching by collate_fn
        '''
    
    def collate_fn(self, batch):
        '''
        Custom collate function to pad variable-length sequences into batches
        
        ⭐ Step A — Find maximum sequence length in batch
            ✔ Iterate through batch items and get max 'seq_length'
            ✔ Determines padding target for this batch
        
        ⭐ Step B — Pad each sequence to max length
            ✔ For each history in batch:
                • Calculate padding_length = max_len - current_length
                • Append PAD_IDX tokens to reach max_len
            ✔ All sequences now have same length
        
        ⭐ Step C — Create attention masks
            ✔ Mask = 1 for real tokens, 0 for padding
            ✔ Prevents model from attending to padding positions
            ✔ Critical for transformer attention mechanism
        
        ⭐ Step D — Collect targets
            ✔ Extract target index for each item in batch
            ✔ No padding needed (targets are single indices)
        
        ⭐ Step E — Convert to PyTorch tensors
            ✔ input_ids: LongTensor of padded histories [batch_size, max_len]
            ✔ targets: LongTensor of target indices [batch_size]
            ✔ attention_mask: BoolTensor for masking [batch_size, max_len]
        
        ⭐ Step F — Return batched dictionary
            ✔ Ready to pass directly to model.forward()
            ✔ Consistent format for training and evaluation
        '''
