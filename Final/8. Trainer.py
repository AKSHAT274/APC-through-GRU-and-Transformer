import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim
import numpy as np
import torch.nn.functional as F

# Device selection: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

class Trainer:
    '''
    Orchestrates training, validation, and checkpointing for GRU and Transformer models
    '''
    def __init__(self, model, train_loader, val_loader, config, model_name):
        '''
        Initialize trainer with model, data loaders, and training configuration
        
        ⭐ Step A — Store core components
            ✔ model: neural network to train (GRU4Rec or Transformer)
            ✔ Move model to device (GPU/CPU) for computation
            ✔ train_loader: DataLoader for training batches
            ✔ val_loader: DataLoader for validation batches
            ✔ config: Config object with hyperparameters
            ✔ model_name: identifier for saving/logging
        
        ⭐ Step B — Configure optimizer
            ✔ AdamW: Adam with weight decay (L2 regularization)
            ✔ lr: config.LEARNING_RATE (initial learning rate)
            ✔ weight_decay: 0.01 (prevents overfitting)
            ✔ betas: (0.9, 0.999) (momentum parameters)
        
        ⭐ Step C — Set up loss function
            ✔ CrossEntropyLoss: for multi-class classification
            ✔ ignore_index=0: don't compute loss on padding tokens
            ✔ label_smoothing: softens hard targets (e.g., 0.1)
            ✔ Improves generalization by preventing overconfidence
        
        ⭐ Step D — Initialize tracking metrics
            ✔ train_losses: list to store training loss per epoch
            ✔ val_losses: list to store validation loss per epoch
            ✔ val_accuracies: list to store validation top-k accuracies
            ✔ best_val_loss: track best validation loss (init: infinity)
            ✔ best_val_acc: track best validation accuracy (init: 0)
            ✔ patience_counter: for early stopping (init: 0)
        
        ⭐ Step E — Configure learning rate scheduling
            ✔ warmup_epochs: gradual LR increase at start (default: 0)
            ✔ CosineAnnealingLR: LR decays following cosine curve
            ✔ T_max: total epochs minus warmup
            ✔ eta_min: minimum LR (1e-6)
            ✔ Helps model converge smoothly
        
        ⭐ Step F — Initialize epoch counter
            ✔ current_epoch: tracks training progress
            ✔ Used for warmup and scheduling decisions
        '''

    def train_epoch(self):
        '''
        Execute one training epoch over all training batches
        
        ⭐ Step A — Set model to training mode
            ✔ self.model.train()
            ✔ Enables dropout and batch normalization training behavior
        
        ⭐ Step B — Initialize loss accumulator
            ✔ total_loss: sum of batch losses
            ✔ Will be averaged at end of epoch
        
        ⭐ Step C — Iterate through batches with progress bar
            ✔ tqdm wrapper for visual progress tracking
            ✔ Shows current batch and loss in real-time
        
        ⭐ Step D — Move batch to device
            ✔ input_ids: [batch, seq_len] of track indices
            ✔ targets: [batch] of ground truth next tracks
            ✔ mask: [batch, seq_len] attention mask for padding
            ✔ Transfer to GPU/CPU as configured
        
        ⭐ Step E — Zero gradients
            ✔ optimizer.zero_grad()
            ✔ Clears previous batch's gradients
            ✔ Prevents gradient accumulation between batches
        
        ⭐ Step F — Forward pass
            ✔ logits = model(input_ids, mask)
            ✔ Returns raw prediction scores [batch, vocab_size]
        
        ⭐ Step G — Compute loss
            ✔ loss = criterion(logits, targets)
            ✔ Compares predictions to ground truth
            ✔ Returns scalar loss value
        
        ⭐ Step H — Backward pass
            ✔ loss.backward()
            ✔ Computes gradients via backpropagation
            ✔ Gradients stored in param.grad for each parameter
        
        ⭐ Step I — Gradient clipping
            ✔ torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ✔ Limits gradient magnitude to prevent exploding gradients
            ✔ Stabilizes training of deep recurrent models
        
        ⭐ Step J — Optimizer step
            ✔ optimizer.step()
            ✔ Updates model parameters using gradients
            ✔ Applies learning rate and weight decay
        
        ⭐ Step K — Accumulate loss
            ✔ total_loss += loss.item()
            ✔ Convert tensor to Python float
            ✔ Update progress bar with current loss
        
        ⭐ Step L — Return average epoch loss
            ✔ total_loss / len(train_loader)
            ✔ Average loss across all batches
        '''
    
    def evaluate(self, data_loader):
        '''
        Evaluate model on validation or test set
        
        ⭐ Step A — Set model to evaluation mode
            ✔ self.model.eval()
            ✔ Disables dropout, uses batch norm in inference mode
        
        ⭐ Step B — Initialize metrics tracking
            ✔ total_loss: accumulator for validation loss
            ✔ top_k_hits: dict counting hits for each k value
            ✔ total_samples: count of evaluated samples
        
        ⭐ Step C — Disable gradient computation
            ✔ with torch.no_grad():
            ✔ Saves memory and speeds up inference
            ✔ Gradients not needed during evaluation
        
        ⭐ Step D — Iterate through batches
            ✔ Loop through data_loader with progress bar
            ✔ Move input_ids, targets, mask to device
        
        ⭐ Step E — Forward pass
            ✔ logits = model(input_ids, mask)
            ✔ Get predictions without computing gradients
        
        ⭐ Step F — Compute loss
            ✔ loss = criterion(logits, targets)
            ✔ Accumulate to total_loss
        
        ⭐ Step G — Calculate top-k predictions
            ✔ torch.topk(logits, max(TOP_K_VALUES), dim=-1)
            ✔ Returns indices of top-k highest scoring tracks
            ✔ Get max k to evaluate all smaller k values
        
        ⭐ Step H — Count top-k hits
            ✔ For each k in TOP_K_VALUES:
                • Extract top-k predictions
                • Check if target in top-k using broadcasting
                • Count hits across batch
            ✔ Accumulate hits for each k value
        
        ⭐ Step I — Count samples
            ✔ total_samples += len(targets)
            ✔ Used for computing final accuracy percentages
        
        ⭐ Step J — Compute average loss
            ✔ avg_loss = total_loss / len(data_loader)
            ✔ Average across all validation batches
        
        ⭐ Step K — Compute accuracies
            ✔ For each k: accuracy = (hits / total_samples) * 100
            ✔ Returns dict: {k: accuracy_percentage}
        
        ⭐ Step L — Return metrics
            ✔ Returns (avg_loss, accuracies)
            ✔ Used for tracking best model and early stopping
        '''
    
    def _warmup_lr(self, epoch):
        '''
        Gradually increase learning rate during warmup period
        
        ⭐ Step A — Check if in warmup phase
            ✔ if epoch < self.warmup_epochs
            ✔ Skip if warmup not configured
        
        ⭐ Step B — Compute warmup scale
            ✔ lr_scale = (epoch + 1) / warmup_epochs
            ✔ Linear ramp from 0 to 1
            ✔ Example: epoch 0/5 → scale 0.2, epoch 4/5 → scale 1.0
        
        ⭐ Step C — Apply scaled learning rate
            ✔ For each param_group in optimizer
            ✔ Set lr = config.LEARNING_RATE * lr_scale
            ✔ Prevents unstable training at start with high LR
        '''

    def train(self):
        '''
        Full training loop with warmup, diagnostics, early stopping, and checkpointing
        
        ⭐ Step A — Print training header
            ✔ Display model name and separator
            ✔ Visual clarity in console output
        
        ⭐ Step B — Loop through epochs
            ✔ for epoch in range(config.NUM_EPOCHS)
            ✔ Update self.current_epoch tracker
        
        ⭐ Step C — Apply learning rate warmup
            ✔ if epoch < warmup_epochs: call _warmup_lr()
            ✔ Print current LR during warmup
            ✔ Stabilizes early training
        
        ⭐ Step D — Post-epoch-1 diagnostics (optional)
            ✔ After first epoch, check training health:
                • Embedding statistics (norms, mean, std)
                • GRU gradient norms (check for vanishing/exploding)
                • Prediction entropy (diversity of outputs)
            ✔ Helps detect issues like mode collapse
            ✔ Warns if entropy too low (model collapsing)
        
        ⭐ Step E — Print epoch info
            ✔ Display current epoch number
        
        ⭐ Step F — Train one epoch
            ✔ train_loss = self.train_epoch()
            ✔ Append to self.train_losses history
        
        ⭐ Step G — Validate
            ✔ val_loss, val_accs = self.evaluate(val_loader)
            ✔ Append to self.val_losses and self.val_accuracies
        
        ⭐ Step H — Print metrics
            ✔ Display train loss, val loss
            ✔ Display top-1, top-5, top-10, top-20 accuracies
            ✔ Display current learning rate
        
        ⭐ Step I — Track best model by Top-10 accuracy
            ✔ if current_val_acc > best_val_acc:
                • Update best_val_acc and best_val_loss
                • Reset patience_counter
                • Save checkpoint as 'best'
                • Print confirmation message
            ✔ else: increment patience_counter
        
        ⭐ Step J — Early stopping check
            ✔ if patience_counter >= config.PATIENCE:
                • Stop training (validation not improving)
                • Print early stopping message
                • Break from epoch loop
        
        ⭐ Step K — Save periodic checkpoints
            ✔ if (epoch + 1) % 5 == 0: save checkpoint
            ✔ Allows recovery if training interrupted
        
        ⭐ Step L — Learning rate scheduling
            ✔ if epoch >= warmup_epochs: scheduler.step()
            ✔ Apply cosine annealing decay
            ✔ Gradually reduces LR for fine-tuning
        
        ⭐ Step M — Return final validation accuracies
            ✔ Returns self.val_accuracies[-1]
            ✔ Dict of top-k accuracies from last epoch
        '''
    
    def save_checkpoint(self, name):
        '''
        Save model checkpoint with training state
        
        ⭐ Step A — Build checkpoint dictionary
            ✔ model_state_dict: model weights and biases
            ✔ optimizer_state_dict: optimizer state (momentum, etc.)
            ✔ train_losses: history of training losses
            ✔ val_losses: history of validation losses
            ✔ val_accuracies: history of validation accuracies
        
        ⭐ Step B — Construct save path
            ✔ path = config.MODEL_DIR / f'{model_name}_{name}.pt'
            ✔ Examples: 'GRU4Rec_best.pt', 'Transformer_epoch_5.pt'
        
        ⭐ Step C — Save to disk
            ✔ torch.save(checkpoint, path)
            ✔ Serializes checkpoint as PyTorch file
            ✔ Can be loaded later for resuming or inference
        '''
