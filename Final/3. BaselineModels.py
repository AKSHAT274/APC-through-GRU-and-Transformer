from config import Config
from typing import Dict, List, Tuple
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class BaselineModels:
    """Traditional ML baselines: KNN and Cosine Similarity"""
    
    def __init__(self, config: Config):
        '''
        Initialize baseline models with configuration and data structures
        
        ✔ Store configuration object for hyperparameters
        ✔ knn_model: placeholder for trained KNN model
        ✔ track_index_to_uri: list mapping matrix indices to track URIs
        ✔ track_uri_to_index: reverse dict for quick URI lookups
        ✔ embedding_matrix: numpy array of all track embeddings [n_tracks, n_features]
        '''
        self.config = config #? To store the configuration object
        self.knn_model = None #? Trained NearestNeighbors model
        self.track_index_to_uri = [] #? List: index → track URI
        self.track_uri_to_index = {} #? Dict: track URI → index
        self.embedding_matrix = None #? Numpy array of embeddings
        
    def train_knn(self, embeddings: Dict) -> NearestNeighbors:
        '''
        Train KNN model on audio feature embeddings
        
        ⭐ Step A — Build embedding matrix from dictionary
            ✔ Extract all track URIs as keys
            ✔ Stack embeddings into numpy matrix [n_tracks, n_features]
            ✔ Ensures consistent ordering for indexing
        
        ⭐ Step B — Create index mappings
            ✔ track_index_to_uri: list of URIs (position = index)
            ✔ track_uri_to_index: dict for O(1) URI → index lookup
            ✔ Store embedding_matrix for later similarity queries
        
        ⭐ Step C — Configure KNN model
            ✔ n_neighbors: config.KNN_NEIGHBORS + 1 (include self match)
            ✔ algorithm='brute': exhaustive search for exact neighbors
            ✔ n_jobs=-1: use all CPU cores for parallel search
        
        ⭐ Step D — Fit KNN on embedding matrix
            ✔ Builds internal index structure
            ✔ Enables fast nearest neighbor queries
            ✔ Store trained model in self.knn_model
        
        ⭐ Step E — Return trained model
            ✔ Model ready for recommendation queries
        '''
    
    def get_knn_recommendations(self, track_uri: str, k: int = 10) -> List[Tuple[str, float]]:
        '''
        Get top-k similar tracks using KNN
        
        ⭐ Step A — Validate input track
            ✔ Check if track_uri exists in vocabulary
            ✔ Return empty list if not found
        
        ⭐ Step B — Lookup track embedding
            ✔ Convert track URI to matrix index
            ✔ Extract embedding vector from embedding_matrix
            ✔ Reshape to [1, n_features] for query
        
        ⭐ Step C — Query KNN model
            ✔ Call knn_model.kneighbors() with k+1 neighbors
            ✔ Returns distances and indices of nearest neighbors
            ✔ +1 because query track itself is included
        
        ⭐ Step D — Exclude self-match
            ✔ Skip the first result (distance=0, query track itself)
            ✔ Take next k neighbors as recommendations
        
        ⭐ Step E — Convert distances to similarities
            ✔ similarity = 1 - distance (for Euclidean distance)
            ✔ Higher similarity = better match
        
        ⭐ Step F — Build recommendation list
            ✔ Map indices back to track URIs
            ✔ Return list of (track_uri, similarity_score) tuples
            ✔ Sorted by similarity (highest first)
        '''
    
    def get_cosine_recommendations(self, track_uri: str, k: int = 10) -> List[Tuple[str, float]]:
        '''
        Get top-k similar tracks using cosine similarity
        
        ⭐ Step A — Validate input track
            ✔ Check if track_uri exists in vocabulary
            ✔ Return empty list if not found
        
        ⭐ Step B — Extract query embedding
            ✔ Convert track URI to matrix index
            ✔ Get embedding vector from embedding_matrix
            ✔ Reshape to [1, n_features] for broadcasting
        
        ⭐ Step C — Compute cosine similarities
            ✔ Use sklearn.cosine_similarity() for efficient computation
            ✔ Computes cosine(query, all_tracks) in one operation
            ✔ Returns similarity scores in range [-1, 1]
        
        ⭐ Step D — Find top-k most similar tracks
            ✔ Use np.argsort() to rank by similarity (descending)
            ✔ Skip first result (self-match with similarity=1.0)
            ✔ Take next k highest scores
        
        ⭐ Step E — Build recommendation list
            ✔ Map top-k indices to track URIs
            ✔ Pair each URI with its similarity score
            ✔ Return list of (track_uri, similarity_score) tuples
        '''
