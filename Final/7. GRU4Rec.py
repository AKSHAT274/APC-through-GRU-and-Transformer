import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU4Rec(nn.Module):
    """GRU-based sequential recommendation model optimized for playlist continuation"""
    
    def __init__(self, num_items, embedding_dim=256, hidden_dim=512, dropout=0.3):
        '''
        
        num_items: vocabulary size (all unique tracks)
        
        
        Initialize GRU4Rec architecture with embeddings, GRU layers, and attention
        
        ⭐ Step A — Define embedding layer
            ✔ num_items: vocabulary size (all unique tracks)
            ✔ embedding_dim: dimensionality of track embeddings (default 256)
            ✔ padding_idx=0: special token for padding, won't be updated
            ✔ Apply lighter dropout (dropout * 0.5) to embeddings
        
        ⭐ Step B — Build 3-layer GRU stack
            ✔ input_size: embedding_dim
            ✔ hidden_size: hidden_dim (default 512)
            ✔ num_layers: 3 (deeper model for complex sequential patterns)
            ✔ batch_first=True: expects input shape [batch, seq_len, features]
            ✔ dropout: between GRU layers for regularization
        
        ⭐ Step C — Add regularization and normalization
            ✔ gru_dropout: dropout after GRU output
            ✔ layer_norm: LayerNorm for stable training
            ✔ Helps with gradient flow and convergence
        
        ⭐ Step D — Define attention mechanism
            ✔ attention_score: Linear layer to compute attention scores
            ✔ Maps hidden_dim → 1 (scalar attention per timestep)
            ✔ Allows model to focus on relevant parts of sequence
        
        ⭐ Step E — Build two-layer prediction head
            ✔ fc1: hidden_dim → hidden_dim // 2 (compression layer)
            ✔ fc1_dropout: dropout before final projection
            ✔ fc2: hidden_dim // 2 → num_items (output logits)
            ✔ Two-layer head improves expressiveness
        
        ⭐ Step F — Weight initialization (optional)
            ✔ Orthogonal initialization for GRU weights
            ✔ Xavier uniform for feed-forward layers
            ✔ Zero initialization for biases
            ✔ Improves training stability and convergence
        '''
    
    def forward(self, x, mask=None):
        '''
        Forward pass: encode sequence with GRU + attention, predict next track
        
        ⭐ Step A — Embed input track IDs
            ✔ x: [batch_size, seq_len] of track indices
            ✔ Pass through embedding layer → [batch, seq_len, embedding_dim]
            ✔ Apply embedding dropout for regularization
        
        ⭐ Step B — Process sequence with GRU
            ✔ Pass embeddings through 3-layer GRU
            ✔ gru_out: [batch, seq_len, hidden_dim] (all timestep outputs)
            ✔ Captures sequential dependencies and ordering patterns
            ✔ Apply GRU dropout to outputs
        
        ⭐ Step C — Compute attention scores
            ✔ Pass gru_out through attention_score layer
            ✔ attention_scores: [batch, seq_len, 1] (score per timestep)
            ✔ Determines importance of each track in history
        
        ⭐ Step D — Apply attention mask
            ✔ If mask provided (for padded sequences):
                • Set attention scores to -inf for padding positions
                • Prevents model from attending to padding tokens
            ✔ Critical for variable-length sequences
        
        ⭐ Step E — Normalize attention with softmax
            ✔ attention_weights: softmax over seq_len dimension
            ✔ Sums to 1.0 across sequence
            ✔ Higher weight = more important timestep
        
        ⭐ Step F — Aggregate sequence with attention
            ✔ Multiply gru_out by attention_weights element-wise
            ✔ Sum over seq_len dimension → context [batch, hidden_dim]
            ✔ Creates weighted representation of entire sequence
            ✔ Uses ALL history, not just last timestep
        
        ⭐ Step G — Normalize context vector
            ✔ Apply LayerNorm to context
            ✔ Stabilizes gradients and training dynamics
        
        ⭐ Step H — Two-layer prediction head
            ✔ fc1 with ReLU activation: compress to hidden_dim // 2
            ✔ Apply fc1_dropout
            ✔ fc2: project to num_items (vocabulary size)
            ✔ Output logits: [batch, num_items]
        
        ⭐ Step I — Return logits
            ✔ Logits represent raw prediction scores for each track
            ✔ Can be converted to probabilities with softmax
            ✔ Used with CrossEntropyLoss during training
        '''
