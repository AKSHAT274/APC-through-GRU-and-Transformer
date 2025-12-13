from config import Config
from typing import List,Dict
from tqdm import tqdm
import json
import pandas as pd
from collections import Counter
class SpotifyDataLoader:
    '''
    Handles loading dataset and merging spotify MPD audio features
    '''
    def __init__(self,config:Config):
        self.config = config #? To store the configuration object
        self.playlist = [] #? To store all the playlists
        self.track_uri_to_id = {} #? map for TrackURI -> IDs
        self.audio_features_df = None #? Features dataframe


    def load_mpd_files(self) -> List[Dict]:
        '''
        1. load_mpd_files() — Loads MPD JSON playlist files
        ➡ Finds all .json files inside the MPD dataset folder.
        
        
        ✔ Load only the first N files
        ✔ Open each JSON file and extract playlists in all_playlists
        ✔ Returns all playlists
        
        '''

    def load_audio_features(self) -> pd.DataFrame:
        '''
        Load CSV of Spotify track features
        '''
    
    def preprocess_data(self,playlists: List[Dict],audio_df: pd.DataFrame):
       '''
       Main cleaning and merging pipeline
       
       ⭐ Step A — Filter playlists by length  - Keeps playlists that have a “reasonable” number of songs.
       ⭐ Step B — Count how often each track URI appears   - Remove rare songs that don’t appear enough times to be useful.
       ⭐ Step C — Keep only frequently occurring tracks    -  Only keep songs that appear in ≥ 10 playlists.
       ⭐ Step D — Build mapping: track URI → track ID 
       ⭐ Step E — Filter playlists again   Keep only songs that: ✔ have audio features  ✔ appear frequently enough
       ⭐ Step F — Build final vocabulary   Collect all unique track URIs:
       ⭐ Step G — Filter audio features to only include valid tracks   -   This ensures your model only sees features for tracks that appear in valid playlists.
       ⭐ Step H — Save the processed data inside the class
       ⭐ Step I — Return final cleaned objects
       
       '''