from sklearn.preprocessing import StandardScaler
from config import Config
from collections import defaultdict, Counter
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

class FeatureEngineer:
    '''
    Create embeddings and transition matrices for baseline models
    '''

    def __init__(self, config: Config):
        '''
        Initialize the FeatureEngineer with configuration and prepare data structures
        
        ✔ Store the configuration object
        ✔ Initialize StandardScaler for feature normalization
        ✔ Create empty dictionary for track embeddings
        ✔ Create empty defaultdict(Counter) for transition matrix
        '''
        self.config = config #? To store the configuration object
        self.scaler = StandardScaler() #? For normalizing audio features
        self.track_embeddings = {} #? Dict: track_uri -> normalized feature vector
        self.transition_matrix = defaultdict(Counter) #? Dict: source_track -> {target_track: count}


    def create_audio_embeddings(self, audio_df: pd.DataFrame, uri_to_id: Dict) -> Dict:
        '''
        Create normalized audio feature embeddings for all tracks
        
        ⭐ Step A — Select relevant audio features from config
            ✔ Extract feature columns specified in config.AUDIO_FEATURES
            ✔ Ensure features exist in the DataFrame
        
        ⭐ Step B — Handle missing values
            ✔ Fill NaN values with median of each feature column
            ✔ Ensures all tracks have complete feature vectors
        
        ⭐ Step C — Normalize features using StandardScaler
            ✔ Fit StandardScaler on feature matrix
            ✔ Transform features to have mean=0 and std=1
            ✔ Improves model convergence and fairness
        
        ⭐ Step D — Build embedding dictionary
            ✔ For each track in audio_df, create URI format: "spotify:track:{id}"
            ✔ Map track URI to its normalized feature vector
            ✔ Store in self.track_embeddings
        
        ⭐ Step E — Return the embeddings dictionary
        '''
    

    def build_transition_matrix(self, playlists: List[Dict]) -> Dict:
        '''
        Build song-to-song transition matrix from playlist sequences
        
        ⭐ Step A — Count transitions between consecutive tracks
            ✔ Iterate through all playlists
            ✔ For each playlist, extract ordered track URIs
            ✔ Count how many times track A is followed by track B
            ✔ Store counts in transition_counts[A][B]
        
        ⭐ Step B — Convert raw counts to probabilities
            ✔ For each source track, sum all transition counts
            ✔ Divide each target count by total to get probability
            ✔ Creates transition_probs[source][target] = P(target|source)
        
        ⭐ Step C — Store and return transition probabilities
            ✔ Save probabilities in self.transition_matrix
            ✔ Used by baseline models for sequential recommendation
            ✔ Returns transition probability dictionary
            
        transition_matrix['spotify:track:6yaxdh87KVj82QIYKTP1zt'] = 
        
        {'spotify:track:4so0Wek9Ig1p6CRCHuINwW': 0.2,
        'spotify:track:0Ie5uiv54KgCr7P4sYDTHl': 0.2,
        'spotify:track:4IdXngKo4g5exqZ0fQTecu': 0.4,
        'spotify:track:6pVW5LRWgeLaHudxauOTJU': 0.2}
        '''
