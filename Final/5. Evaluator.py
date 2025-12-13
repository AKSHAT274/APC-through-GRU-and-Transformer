from collections import defaultdict
from config import Config
from typing import List
import pandas as pd
from tqdm import tqdm


class Evaluator:
    """Evaluate baseline and deep learning models on recommendation tasks"""
    
    def __init__(self, config: Config):
        '''
        Initialize the Evaluator with configuration and results storage
        
        ✔ Store configuration object for accessing evaluation settings
        ✔ results: defaultdict(dict) to store metrics for each model
        ✔ Structure: results[model_name][metric] = value
        ✔ Allows easy comparison across models
        '''
        self.config = config #? To store the configuration object
        self.results = defaultdict(dict) #? Dict: model_name -> {metric: value}
        
    def evaluate_top_k_accuracy(self, predictions: List[str], target: str, k: int) -> int:
        '''
        Check if target track is in top-k predictions
        
        ⭐ Step A — Extract top-k predictions
            ✔ predictions[:k]: get first k predicted tracks
            ✔ predictions list is already sorted by relevance/score
        
        ⭐ Step B — Check if target is present
            ✔ Return 1 if target in top-k (hit)
            ✔ Return 0 if target not in top-k (miss)
        
        ⭐ Step C — Use case
            ✔ Binary metric: did we succeed or not?
            ✔ Aggregate hits across test set to compute accuracy
            ✔ Top-k accuracy = (total_hits / total_samples) * 100
        '''
    
    def evaluate_baseline(self, model, test_df: pd.DataFrame, model_name: str):
        '''
        Evaluate a baseline model (KNN or Cosine) on the test set
        
        ⭐ Step A — Initialize metrics tracking
            ✔ top_k_hits: dict to count hits for each k value
            ✔ k values from config.TOP_K_VALUES (e.g., [5, 10, 20])
            ✔ total: counter for valid test samples processed
        
        ⭐ Step B — Iterate through test sequences
            ✔ For each row in test_df:
                • Extract 'history' (list of tracks) and 'target' (ground truth)
                • Skip if history is empty (can't make prediction)
        
        ⭐ Step C — Get last track as query
            ✔ last_track = history[-1]
            ✔ Most recent track in playlist
            ✔ Used as seed for finding similar tracks
        
        ⭐ Step D — Generate recommendations
            ✔ If model_name == "KNN": call get_knn_recommendations()
            ✔ If model_name == "Cosine": call get_cosine_recommendations()
            ✔ Request max(TOP_K_VALUES) recommendations (e.g., 20)
            ✔ Returns list of (track_uri, score) tuples
        
        ⭐ Step E — Handle empty recommendations
            ✔ If model returns no recommendations, skip sample
            ✔ Happens when last_track not in vocabulary
        
        ⭐ Step F — Extract predicted track URIs
            ✔ predicted_tracks = [uri for uri, score in recs]
            ✔ Strip scores, keep only track identifiers
            ✔ Sorted by relevance (highest scores first)
        
        ⭐ Step G — Calculate top-k accuracy for all k values
            ✔ For each k in config.TOP_K_VALUES:
                • Call evaluate_top_k_accuracy(predicted_tracks, target, k)
                • Increment top_k_hits[k] if hit
            ✔ Tracks multiple metrics simultaneously
        
        ⭐ Step H — Increment total counter
            ✔ Count this as a valid evaluation sample
            ✔ Used for computing final accuracy percentages
        
        ⭐ Step I — Optional: limit evaluation samples
            ✔ if total >= 10000: break
            ✔ For faster evaluation during development/debugging
            ✔ Remove limit for full evaluation
        
        ⭐ Step J — Compute final accuracies
            ✔ For each k: accuracy = (hits / total) * 100
            ✔ Converts raw counts to percentages
            ✔ Store in self.results[model_name]
        
        ⭐ Step K — Print results
            ✔ Display model name and sample count
            ✔ Print Top-k accuracy for each k value
            ✔ Formatted as percentages with 2 decimal places
        
        ⭐ Step L — Return accuracies dictionary
            ✔ Returns {k: accuracy} for further analysis
            ✔ Stored in self.results for comparison
        '''
