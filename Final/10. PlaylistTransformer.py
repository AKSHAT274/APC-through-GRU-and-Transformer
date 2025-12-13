import torch
import torch.nn as nn
import math
from PositionalEncoding import PositionalEncoding

class PlaylistTransformer(nn.Module):
    """
    Enhanced Transformer architecture for playlist continuation and next-track prediction
    """
    def __init__(self, num_items, d_model=256, nhead=8, num_layer=4, dropout=0.1):
        '''
        Initialize Transformer architecture with embeddings, encoder layers, and prediction head
        
        ⭐ Step A — Define token embedding layer
            ✔ num_items: vocabulary size (all unique tracks + special tokens)
            ✔ d_model: embedding dimension (default 256)
            ✔ padding_idx=0: padding token won't be updated during training
            ✔ Converts track IDs to dense vectors
        
        ⭐ Step B — Add positional encoding
            ✔ pos_encoder: PositionalEncoding module
            ✔ Injects position information into embeddings
            ✔ Uses sinusoidal patterns from PositionalEncoding class
            ✔ Includes dropout for regularization
        
        ⭐ Step C — Configure Transformer encoder layer
            ✔ d_model: hidden dimension for all sublayers
            ✔ nhead: number of attention heads (default 8)
            ✔ dim_feedforward: d_model * 4 (wider FFN for expressiveness)
            ✔ dropout: regularization probability
            ✔ batch_first=True: expects [batch, seq_len, d_model]
            ✔ norm_first=True: Pre-LN architecture (better gradient flow)
        
        ⭐ Step D — Build Transformer encoder stack
            ✔ encoder_layer: single transformer layer (attention + FFN)
            ✔ num_layers: stack multiple layers (default 4)
            ✔ norm: final LayerNorm for output stabilization
            ✔ Creates deep network for complex pattern learning
        
        ⭐ Step E — Build two-layer prediction head
            ✔ dropout: regularization before prediction
            ✔ fc1: d_model → d_model (intermediate layer)
            ✔ activation: GELU (smooth, non-linear activation)
            ✔ fc2: d_model → num_items (output logits)
            ✔ Two-layer MLP improves prediction quality
        
        ⭐ Step F — Store model dimension
            ✔ self.d_model: used for embedding scaling
            ✔ Standard practice in Transformer architectures
        
        ⭐ Step G — Initialize weights
            ✔ Call _init_weights() for better training start
            ✔ Proper initialization prevents gradient issues
        '''

    def _init_weights(self):
        '''
        Initialize model weights for stable training
        
        ⭐ Step A — Initialize embedding weights
            ✔ nn.init.normal_(embedding.weight, mean=0, std=0.02)
            ✔ Small standard deviation for stable gradients
            ✔ Prevents large initial embeddings
        
        ⭐ Step B — Zero out padding embedding
            ✔ nn.init.constant_(embedding.weight[0], 0)
            ✔ PAD token (index 0) remains zero vector
            ✔ Ensures padding doesn't contribute to predictions
        
        ⭐ Step C — Initialize linear layers
            ✔ Xavier uniform initialization for fc1 and fc2 weights
            ✔ Maintains variance across layers
            ✔ Zero initialization for biases
            ✔ Standard practice for feedforward layers
        '''

    def forward(self, x, src_key_padding_mask=None):
        '''
        Forward pass: encode sequence and predict next track
        
        Args:
            x: Input tensor [batch_size, seq_len] of track indices
            src_key_padding_mask: Boolean mask [batch, seq_len], True for valid tokens
        
        Returns:
            logits: Output predictions [batch_size, num_items]
        
        ⭐ Step A — Embed input tokens
            ✔ emb = self.embedding(x)
            ✔ Converts indices to vectors: [batch, seq_len, d_model]
            ✔ Scale embeddings by sqrt(d_model)
            ✔ Scaling helps maintain gradient magnitude
        
        ⭐ Step B — Add positional encoding
            ✔ emb = self.pos_encoder(emb)
            ✔ Injects position information via sinusoidal patterns
            ✔ Dropout applied inside pos_encoder
            ✔ Model now knows token order
        
        ⭐ Step C — Invert padding mask
            ✔ PyTorch TransformerEncoder expects True for positions to IGNORE
            ✔ Input mask has True for VALID tokens
            ✔ Invert: src_key_padding_mask = ~src_key_padding_mask
            ✔ Now True = padding, False = valid
        
        ⭐ Step D — Process through Transformer encoder
            ✔ out = self.transformer(emb, src_key_padding_mask)
            ✔ Multi-head self-attention captures dependencies
            ✔ Feed-forward networks add non-linearity
            ✔ Output: [batch, seq_len, d_model] (all positions encoded)
        
        ⭐ Step E — Extract last token representation
            ✔ last_output = out[:, -1, :]
            ✔ Use final position for next-track prediction
            ✔ Shape: [batch, d_model]
            ✔ Contains context from entire sequence
        
        ⭐ Step F — First layer of prediction head
            ✔ hidden = self.fc1(last_output)
            ✔ Project to intermediate space
            ✔ Still d_model dimensions
        
        ⭐ Step G — Apply activation
            ✔ hidden = self.activation(hidden)
            ✔ GELU: Gaussian Error Linear Unit
            ✔ Smoother than ReLU, works better for Transformers
        
        ⭐ Step H — Apply dropout
            ✔ hidden = self.dropout(hidden)
            ✔ Regularization before final prediction
            ✔ Prevents overfitting
        
        ⭐ Step I — Project to vocabulary
            ✔ logits = self.fc2(hidden)
            ✔ Shape: [batch, num_items]
            ✔ Raw scores for each track in vocabulary
        
        ⭐ Step J — Return logits
            ✔ Used with CrossEntropyLoss during training
            ✔ Convert to probabilities with softmax for inference
        '''

    def get_attention_weights(self, x, src_key_padding_mask=None):
        '''
        Extract attention weights from all encoder layers for visualization
        
        Args:
            x: Input tensor [batch_size, seq_len]
            src_key_padding_mask: Optional padding mask
        
        Returns:
            attention_weights: List of attention matrices from each layer
        
        ⭐ Step A — Prepare embeddings
            ✔ Embed tokens: emb = embedding(x) * sqrt(d_model)
            ✔ Add positional encoding: emb = pos_encoder(emb)
            ✔ Same preprocessing as forward pass
        
        ⭐ Step B — Invert padding mask
            ✔ if mask provided: invert to PyTorch format
            ✔ True = padding, False = valid
        
        ⭐ Step C — Initialize storage for attention weights
            ✔ attention_weights: empty list
            ✔ Will collect attention matrices from each layer
        
        ⭐ Step D — Define hook function
            ✔ hook_fn captures attention weights during forward pass
            ✔ output[1] contains attention weights from self_attn module
            ✔ Detach and store to prevent gradient tracking
            ✔ Hook intercepts computation without modifying it
        
        ⭐ Step E — Register hooks on all layers
            ✔ Loop through transformer.layers
            ✔ Register forward hook on each layer.self_attn module
            ✔ Store handles for later removal
            ✔ Hooks will fire during forward pass
        
        ⭐ Step F — Run forward pass with no_grad
            ✔ with torch.no_grad(): disable gradient computation
            ✔ Call transformer(emb, mask)
            ✔ Hooks capture attention weights during execution
            ✔ No need to compute predictions, just attention
        
        ⭐ Step G — Clean up hooks
            ✔ Loop through handles and call handle.remove()
            ✔ Prevents hooks from firing in future forward passes
            ✔ Good practice to avoid memory leaks
        
        ⭐ Step H — Return attention weights
            ✔ List of attention matrices, one per layer
            ✔ Each matrix: [batch, nhead, seq_len, seq_len]
            ✔ Shows which positions attend to which positions
            ✔ Can be visualized as heatmaps for interpretability
        '''
