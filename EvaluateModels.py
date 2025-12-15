import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from TraningConfig import TrainingConfig
from PlayListDataset import PlaylistDataset
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


def r_precision_at_R(recs: np.ndarray, target_idx: int, R: int) -> float:
    # R-precision where relevant set size = R. In our dataset R=1 (single target).
    R = max(1, R)
    top_R = recs[:R]
    hits = (top_R == target_idx).sum()
    return float(hits) / float(R)


def ndcg_at_k(recs: np.ndarray, target_idx: int, k: int) -> float:
    top_k = recs[:k]
    # Find rank (0-based) of target
    match_positions = np.where(top_k == target_idx)[0]
    if match_positions.size == 0:
        return 0.0
    rank = int(match_positions[0])
    dcg = 1.0 / math.log2(rank + 2)  # rank 0 -> 1/log2(2) = 1
    idcg = 1.0  # only one relevant item
    return dcg / idcg


def clicks_at_500(recs: np.ndarray, target_idx: int) -> int:
    # 10 items per page; first 10 need 0 clicks. Max 50 pages -> 0..49 clicks, else 51.
    top_500 = recs[:500]
    match_positions = np.where(top_500 == target_idx)[0]
    if match_positions.size == 0:
        return 51
    rank = int(match_positions[0])  # 0-based index in the 500 list
    clicks = rank // 10
    return clicks


def evaluate_model(model, data_loader, num_items: int, top_k_eval: int = 500, device: torch.device = torch.device('cpu')):
    model.eval()
    rprec_list = []
    ndcg_list = []
    clicks_list = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            mask = batch['attention_mask'].to(device)

            logits = model(input_ids, mask)
            # Get top-k indices
            _, top_indices = torch.topk(logits, k=min(top_k_eval, num_items), dim=-1)

            # Compute metrics per sample
            for i in range(top_indices.size(0)):
                recs = top_indices[i].detach().cpu().numpy()
                target_idx = int(targets[i].item())

                rprec = r_precision_at_R(recs, target_idx, R=1)
                ndcg = ndcg_at_k(recs, target_idx, k=top_k_eval)
                clicks = clicks_at_500(recs, target_idx)

                rprec_list.append(rprec)
                ndcg_list.append(ndcg)
                clicks_list.append(clicks)

    metrics = {
        'R-Prec@R': float(np.mean(rprec_list)) if rprec_list else 0.0,
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

    config = TrainingConfig()

    # Build vocabulary from train set to match training
    vocab = build_vocabulary(str(config.DATA_DIR / 'train.parquet'))
    num_items = len(vocab) + 2  # PAD + UNK

    # Datasets and loaders
    test_dataset = PlaylistDataset(
        str(config.DATA_DIR / 'test.parquet'),
        vocab,
        config.MAX_SEQ_LENGTH,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=config.NUM_WORKERS,
    )

    # Models
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
    gru_ckpt = torch.load(gru_ckpt_path, map_location=device)
    trans_ckpt = torch.load(trans_ckpt_path, map_location=device)
    gru_model.load_state_dict(gru_ckpt['model_state_dict'])
    transformer_model.load_state_dict(trans_ckpt['model_state_dict'])

    # Evaluate
    print("\nEvaluating GRU on test set (R-Prec, NDCG, Clicks@500)...")
    gru_metrics = evaluate_model(gru_model, test_loader, num_items, top_k_eval=500, device=device)
    print({k: (round(v, 4) if k != 'Clicks@500' else round(v, 2)) for k, v in gru_metrics.items()})

    print("\nEvaluating Transformer on test set (R-Prec, NDCG, Clicks@500)...")
    trans_metrics = evaluate_model(transformer_model, test_loader, num_items, top_k_eval=500, device=device)
    print({k: (round(v, 4) if k != 'Clicks@500' else round(v, 2)) for k, v in trans_metrics.items()})

    # Optional: save metrics
    out = pd.DataFrame([
        {'model': 'GRU', **gru_metrics},
        {'model': 'Transformer', **trans_metrics},
    ])
    out_path = config.OUTPUT_DIR / 'metrics' / 'dl_rank_metrics.csv'
    out.to_csv(out_path, index=False)
    print(f"\nSaved metrics to {out_path}")


if __name__ == '__main__':
    main()
