from config import Config
from typing import List, Dict, Tuple
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path

class TransformerDataPreparation:
    """Prepare sequential data for Transformer training"""
    
    def __init__(self, config: Config):
        '''
        Initialize the TransformerDataPreparation with configuration
        
        ✔ Store the configuration object for accessing hyperparameters
        ✔ Configuration includes sequence length, split ratios, batch size
        '''
        self.config = config #? To store the configuration object
        
    def create_sequences(self, playlists: List[Dict], uri_to_id: Dict) -> pd.DataFrame:
        '''
        Create sliding window sequences for next-song prediction
        
        ⭐ Step A — Extract playlist metadata and tracks
            ✔ For each playlist, get playlist ID, name, and ordered track list
            ✔ Track URIs represent the sequential order of songs
        
        ⭐ Step B — Generate training sequences using sliding window
            ✔ For position i in playlist, create sequence:
                • history = tracks[0:i] (all previous tracks)
                • target = tracks[i] (next track to predict)
            ✔ Starts from position 1 (need at least 1 history track)
            ✔ Each position creates one training example
        
        ⭐ Step C — Store sequence metadata
            ✔ Save playlist_id to track data provenance
            ✔ Save playlist_name for context-aware features
            ✔ Save position in playlist and total playlist length
            ✔ Helps model learn positional patterns
        
        ⭐ Step D — Convert to DataFrame
            ✔ Each row = one training sequence
            ✔ Columns: playlist_id, playlist_name, history, target, position, playlist_length
            ✔ Returns DataFrame ready for train/val/test splitting
        '''
    
    def train_val_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        Split data ensuring no playlist overlap between train/val/test sets
        
        ⭐ Step A — Extract and shuffle unique playlist IDs
            ✔ Get all unique playlist IDs from the DataFrame
            ✔ Shuffle playlists randomly using config.RANDOM_SEED
            ✔ Ensures reproducible splits
        
        ⭐ Step B — Calculate split boundaries
            ✔ Compute train split: config.TRAIN_RATIO of total playlists
            ✔ Compute val split: config.VAL_RATIO of total playlists
            ✔ Remaining playlists go to test set
            ✔ Typical ratios: 70% train, 15% val, 15% test
        
        ⭐ Step C — Assign playlists to sets
            ✔ First n_train playlists → train_playlists
            ✔ Next n_val playlists → val_playlists
            ✔ Remaining playlists → test_playlists
            ✔ No playlist appears in multiple sets
        
        ⭐ Step D — Filter DataFrame by playlist assignment
            ✔ train_df = all sequences from train_playlists
            ✔ val_df = all sequences from val_playlists
            ✔ test_df = all sequences from test_playlists
            ✔ Prevents data leakage across splits
        
        ⭐ Step E — Return three DataFrames
            ✔ Returns (train_df, val_df, test_df) tuple
            ✔ Ready for separate dataloaders
        '''
    
    def save_to_arrow(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                      test_df: pd.DataFrame, output_dir: Path):
        '''
        Save datasets as Arrow/Parquet files for efficient loading
        
        ⭐ Step A — Convert list columns to string format
            ✔ 'history' column contains Python lists of track URIs
            ✔ Arrow/Parquet doesn't natively handle Python lists
            ✔ Create 'history_str' column: join tracks with '|' delimiter
            ✔ Example: ['track1', 'track2'] → 'track1|track2'
        
        ⭐ Step B — Save DataFrames as Parquet files
            ✔ Save train_df to output_dir/data/train.parquet
            ✔ Save val_df to output_dir/data/val.parquet
            ✔ Save test_df to output_dir/data/test.parquet
            ✔ Parquet uses Apache Arrow format internally
        
        ⭐ Step C — Benefits of Parquet format
            ✔ Fast columnar reads (only load needed columns)
            ✔ Efficient compression (smaller file sizes)
            ✔ Native pandas integration
            ✔ Much faster than CSV for large datasets
        '''
