import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer models
    Adds position information to token embeddings
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        '''
        Initialize positional encoding with sinusoidal patterns
        
        ⭐ Step A — Store dropout layer
            ✔ dropout: nn.Dropout with probability p
            ✔ Applied after adding positional encodings
            ✔ Regularizes the model during training
        
        ⭐ Step B — Create position indices
            ✔ position: tensor [0, 1, 2, ..., max_len-1]
            ✔ unsqueeze(1): reshape to [max_len, 1] for broadcasting
            ✔ Represents absolute position in sequence
        
        ⭐ Step C — Compute frequency division term
            ✔ div_term: exponential decay for different dimensions
            ✔ Formula: exp(arange(0, d_model, 2) * (-log(10000) / d_model))
            ✔ Creates different wavelengths for each dimension pair
            ✔ Lower dimensions = higher frequency, higher = lower frequency
        
        ⭐ Step D — Initialize positional encoding matrix
            ✔ pe: zeros tensor of shape [max_len, d_model]
            ✔ Will store sinusoidal patterns for all positions
        
        ⭐ Step E — Fill even dimensions with sine
            ✔ pe[:, 0::2] = sin(position * div_term)
            ✔ Even indices (0, 2, 4, ...) use sine function
            ✔ Creates smooth periodic patterns
        
        ⭐ Step F — Fill odd dimensions with cosine
            ✔ pe[:, 1::2] = cos(position * div_term)
            ✔ Odd indices (1, 3, 5, ...) use cosine function
            ✔ Cosine is 90° phase-shifted from sine
            ✔ Together, sine/cosine provide unique encoding for each position
        
        ⭐ Step G — Register as buffer (not trainable)
            ✔ self.register_buffer('pe', pe)
            ✔ Stores pe as part of model state but not as parameter
            ✔ Won't be updated during backpropagation
            ✔ Will be moved to GPU along with model
        '''

    def forward(self, x):
        '''
        Add positional encoding to input embeddings
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns:
            Tensor of shape [batch_size, seq_len, d_model] with positional info
        
        ⭐ Step A — Extract sequence length
            ✔ seq_len = x.size(1)
            ✔ Determines how many positional encodings to use
            ✔ Handles variable-length sequences
        
        ⭐ Step B — Slice positional encodings
            ✔ self.pe[:seq_len]: get first seq_len positions [seq_len, d_model]
            ✔ unsqueeze(0): add batch dimension [1, seq_len, d_model]
            ✔ Prepares for broadcasting across batch
        
        ⭐ Step C — Add positional encoding to embeddings
            ✔ x = x + self.pe[:seq_len].unsqueeze(0)
            ✔ Broadcasting: [batch, seq_len, d_model] + [1, seq_len, d_model]
            ✔ Each position gets same positional encoding across batch
            ✔ Injects absolute position information into embeddings
        
        ⭐ Step D — Apply dropout
            ✔ self.dropout(x): randomly zeros some elements
            ✔ Prevents overfitting to exact positions
            ✔ Returns final encoded tensor
        
        ⭐ Step E — Why this works
            ✔ Sinusoidal patterns are deterministic (not learned)
            ✔ Different frequencies allow model to learn relative positions
            ✔ Model can extrapolate to longer sequences than seen in training
            ✔ Essential for Transformer to distinguish token order
        '''
