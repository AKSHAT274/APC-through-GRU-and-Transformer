import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from TrainingConfig import TrainingConfig
from GRU import GRU
from PlaylistTransformer import PlaylistTransformer


def build_vocabulary(train_path: str):
    df = pd.read_parquet(train_path)
    all_tracks = set()
    for _, row in df.iterrows():
        if isinstance(row['history'], str):
            tracks = row['history'].split('|') if row['history'] else []
        else:
            tracks = row['history']
        all_tracks.update(tracks)
        all_tracks.add(row['target'])
    vocab = sorted(list(all_tracks))
    return vocab


def reconstruct_playlists(test_df: pd.DataFrame):
    """Reconstruct full playlists from sequence data"""
    playlists = []
    
    for pid in test_df['playlist_id'].unique():
        playlist_rows = test_df[test_df['playlist_id'] == pid]
        # Get the row with maximum position (has full history)
        max_pos_row = playlist_rows.loc[playlist_rows['position'].idxmax()]
        
        # Extract full track list
        if isinstance(max_pos_row['history'], str):
            history = max_pos_row['history'].split('|') if max_pos_row['history'] else []
        else:
            history = max_pos_row['history']
        
        full_tracks = history + [max_pos_row['target']]
        
        playlists.append({
            'playlist_id': pid,
            'tracks': full_tracks,
            'playlist_name': max_pos_row.get('playlist_name', ''),
        })
    
    return playlists


def r_precision(ground_truth_set: set, recommended_list: list) -> float:
    """R-Precision: |G ∩ R[:len(G)]| / |G|"""
    R = len(ground_truth_set)
    if R == 0:
        return 0.0
    top_R = recommended_list[:R]
    hits = len(ground_truth_set.intersection(set(top_R)))
    return float(hits) / float(R)


def ndcg_at_k(ground_truth_set: set, recommended_list: list, k: int) -> float:
    """NDCG@k for multiple relevant items"""
    if len(ground_truth_set) == 0:
        return 0.0
    
    top_k = recommended_list[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, track in enumerate(top_k):
        if track in ground_truth_set:
            dcg += 1.0 / math.log2(i + 2)  # i is 0-based, so rank is i+1, log2(rank+1)
    
    # Calculate IDCG (ideal: all relevant items at top)
    idcg = 0.0
    for i in range(min(len(ground_truth_set), k)):
        idcg += 1.0 / math.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def clicks_at_500(ground_truth_set: set, recommended_list: list) -> int:
    """Recommended Songs Clicks: number of refreshes (pages of 10) needed"""
    top_500 = recommended_list[:500]
    
    # Find first relevant track
    for i, track in enumerate(top_500):
        if track in ground_truth_set:
            clicks = i // 10  # 0-9 -> 0 clicks, 10-19 -> 1 click, etc.
            return clicks
    
    return 51  # No relevant track found


def evaluate_model_on_playlists(model, playlists, track_to_idx, idx_to_track, num_items: int, 
                                  max_seq_len: int = 50, device: torch.device = torch.device('cpu')):
    """Evaluate model on complete playlists with 60% holdout"""
    model.eval()
    
    PAD_IDX = 0
    UNK_IDX = 1
    
    rprec_list = []
    ndcg_list = []
    clicks_list = []
    
    with torch.no_grad():
        for playlist in tqdm(playlists, desc="Evaluating playlists"):
            tracks = playlist['tracks']
            
            if len(tracks) < 5:  # Skip very short playlists
                continue
            
            # Split: 40% visible, 60% hidden
            split_point = int(len(tracks) * 0.4)
            split_point = max(1, split_point)  # At least 1 track visible
            
            visible_tracks = tracks[:split_point]
            hidden_tracks = tracks[split_point:]
            
            if len(hidden_tracks) == 0:
                continue
            
            # Convert visible tracks to indices
            visible_indices = [track_to_idx.get(t, UNK_IDX) for t in visible_tracks]
            
            # Truncate if too long
            if len(visible_indices) > max_seq_len:
                visible_indices = visible_indices[-max_seq_len:]
            
            # Pad to max_seq_len
            padding_length = max_seq_len - len(visible_indices)
            padded_input = visible_indices + [PAD_IDX] * padding_length
            mask = [1] * len(visible_indices) + [0] * padding_length
            
            # Convert to tensors
            input_ids = torch.LongTensor([padded_input]).to(device)
            attention_mask = torch.BoolTensor([mask]).to(device)
            
            # Get recommendations
            logits = model(input_ids, attention_mask)
            _, top_indices = torch.topk(logits[0], k=min(500, num_items), dim=-1)
            
            # Convert indices back to track URIs
            recommended_tracks = []
            for idx in top_indices.cpu().numpy():
                if idx in idx_to_track:
                    track = idx_to_track[idx]
                    # Don't recommend tracks already in visible set
                    if track not in visible_tracks:
                        recommended_tracks.append(track)
                if len(recommended_tracks) >= 500:
                    break
            
            # Ground truth set
            ground_truth_set = set(hidden_tracks)
            
            # Calculate metrics
            rprec = r_precision(ground_truth_set, recommended_tracks)
            ndcg = ndcg_at_k(ground_truth_set, recommended_tracks, k=500)
            clicks = clicks_at_500(ground_truth_set, recommended_tracks)
            
            rprec_list.append(rprec)
            ndcg_list.append(ndcg)
            clicks_list.append(clicks)
    
    metrics = {
        'R-Precision': float(np.mean(rprec_list)) if rprec_list else 0.0,
        'NDCG@500': float(np.mean(ndcg_list)) if ndcg_list else 0.0,
        'Clicks@500': float(np.mean(clicks_list)) if clicks_list else 51.0,
    }
    
    return metrics


def main():
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")
    
    config = TrainingConfig()

    # Build vocabulary from train set to match training
    print("\n[1/5] Building vocabulary from training data...")
    vocab = build_vocabulary(str(config.DATA_DIR / 'train.parquet'))
    num_items = len(vocab) + 2  # PAD + UNK
    print(f"Vocabulary size: {len(vocab):,} tracks, num_items: {num_items:,}")
    
    # Build track mappings
    PAD_IDX = 0
    UNK_IDX = 1
    track_to_idx = {track: idx + 2 for idx, track in enumerate(vocab)}
    idx_to_track = {idx: track for track, idx in track_to_idx.items()}

    # Load and reconstruct test playlists
    print("\n[2/5] Reconstructing test playlists...")
    test_df = pd.read_parquet(str(config.DATA_DIR / 'test.parquet'))
    test_playlists = reconstruct_playlists(test_df)
    print(f"Reconstructed {len(test_playlists):,} test playlists")

    # Models
    print("\n[3/5] Loading models...")
    gru_model = GRU(
        num_items=num_items,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT,
    ).to(device)

    transformer_model = PlaylistTransformer(
        num_items=num_items,
        d_model=config.EMBEDDING_DIM,
        nhead=config.NUM_HEADS,
        num_layer=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)

    # Load best checkpoints
    gru_ckpt_path = config.MODEL_DIR / 'GRU_best.pt'
    trans_ckpt_path = config.MODEL_DIR / 'Transformer_best.pt'
    
    print(f"Loading GRU from: {gru_ckpt_path}")
    print(f"Loading Transformer from: {trans_ckpt_path}")
    
    gru_ckpt = torch.load(gru_ckpt_path, map_location=device)
    trans_ckpt = torch.load(trans_ckpt_path, map_location=device)
    gru_model.load_state_dict(gru_ckpt['model_state_dict'])
    transformer_model.load_state_dict(trans_ckpt['model_state_dict'])

    # Evaluate
    print("\n[4/5] Evaluating GRU on test playlists (60% holdout)...")
    print("Computing R-Precision, NDCG@500, Clicks@500...")
    gru_metrics = evaluate_model_on_playlists(
        gru_model, test_playlists, track_to_idx, idx_to_track, 
        num_items, config.MAX_SEQ_LENGTH, device
    )
    
    print("\n[5/5] Evaluating Transformer on test playlists (60% holdout)...")
    print("Computing R-Precision, NDCG@500, Clicks@500...")
    trans_metrics = evaluate_model_on_playlists(
        transformer_model, test_playlists, track_to_idx, idx_to_track,
        num_items, config.MAX_SEQ_LENGTH, device
    )

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS (60% holdout per playlist)")
    print("=" * 80)
    
    print("\nGRU:")
    for k, v in gru_metrics.items():
        if k == 'Clicks@500':
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v:.4f}")
    
    print("\nTransformer:")
    for k, v in trans_metrics.items():
        if k == 'Clicks@500':
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v:.4f}")

    # Save metrics
    out = pd.DataFrame([
        {'model': 'GRU', **gru_metrics},
        {'model': 'Transformer', **trans_metrics},
    ])
    out_path = config.OUTPUT_DIR / 'metrics' / 'playlist_continuation_metrics.csv'
    out.to_csv(out_path, index=False)
    print(f"\nSaved metrics to {out_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
