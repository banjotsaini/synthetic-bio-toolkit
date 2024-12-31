"""
Feature engineering module for converting DNA sequences into numerical representations.
"""

import numpy as np
from typing import List, Dict, Union, Optional
from collections import Counter
import itertools
from Bio.Seq import Seq
import pandas as pd

class SequenceFeatureGenerator:
    def __init__(self):
        """Initialize the feature generator with default settings."""
        self.nucleotides = ['A', 'C', 'G', 'T']
        # Generate all possible k-mers for k=1,2,3
        self.kmers = {
            1: [''.join(p) for p in itertools.product(self.nucleotides, repeat=1)],
            2: [''.join(p) for p in itertools.product(self.nucleotides, repeat=2)],
            3: [''.join(p) for p in itertools.product(self.nucleotides, repeat=3)]
        }

    def one_hot_encode(self, sequence: str) -> np.ndarray:
        """
        Convert DNA sequence to one-hot encoding.
        
        Args:
            sequence (str): DNA sequence
            
        Returns:
            np.ndarray: One-hot encoded sequence (length x 4)
        """
        # Create mapping of nucleotides to indices
        nuc_to_idx = {nuc: idx for idx, nuc in enumerate(self.nucleotides)}
        
        # Initialize encoding matrix
        encoding = np.zeros((len(sequence), len(self.nucleotides)))
        
        # Fill in encoding
        for i, nuc in enumerate(sequence):
            if nuc in nuc_to_idx:
                encoding[i, nuc_to_idx[nuc]] = 1
                
        return encoding

    def kmer_frequencies(self, sequence: str, k: int = 3) -> Dict[str, float]:
        """
        Calculate k-mer frequencies for a sequence.
        
        Args:
            sequence (str): DNA sequence
            k (int): Length of k-mers
            
        Returns:
            Dict[str, float]: Dictionary of k-mer frequencies
        """
        if k not in self.kmers:
            raise ValueError(f"k={k} not supported. Use k=1,2,3")
            
        # Count k-mers
        kmers = [''.join(seq) for seq in zip(*[sequence[i:] for i in range(k)])]
        kmer_counts = Counter(kmers)
        
        # Calculate frequencies
        total_kmers = len(kmers)
        frequencies = {kmer: kmer_counts.get(kmer, 0) / total_kmers 
                      for kmer in self.kmers[k]}
                      
        return frequencies

    def physical_properties(self, sequence: str) -> Dict[str, float]:
        """
        Calculate physical properties of the DNA sequence.
        
        Args:
            sequence (str): DNA sequence
            
        Returns:
            Dict[str, float]: Dictionary of physical properties
        """
        seq_obj = Seq(sequence)
        
        properties = {
            'gc_content': (sequence.count('G') + sequence.count('C')) / len(sequence),
            'length': len(sequence),
            'weight': len(sequence) * 660,  # Approximate molecular weight in Da
            'gc_skew': ((sequence.count('G') - sequence.count('C')) / 
                       (sequence.count('G') + sequence.count('C'))
                       if (sequence.count('G') + sequence.count('C')) > 0 else 0)
        }
        
        return properties

    def generate_all_features(self, sequence: str) -> Dict[str, Union[float, Dict]]:
        """
        Generate all features for a sequence.
        
        Args:
            sequence (str): DNA sequence
            
        Returns:
            Dict: Dictionary containing all features
        """
        features = {
            'physical': self.physical_properties(sequence),
            'kmer_1': self.kmer_frequencies(sequence, k=1),
            'kmer_2': self.kmer_frequencies(sequence, k=2),
            'kmer_3': self.kmer_frequencies(sequence, k=3)
        }
        
        return features

    def features_to_vector(self, features: Dict) -> np.ndarray:
        """
        Convert features dictionary to flat feature vector.
        
        Args:
            features (Dict): Features dictionary from generate_all_features
            
        Returns:
            np.ndarray: Flat feature vector
        """
        vector = []
        
        # Add physical properties
        vector.extend([
            features['physical']['gc_content'],
            features['physical']['length'],
            features['physical']['weight'],
            features['physical']['gc_skew']
        ])
        
        # Add k-mer frequencies in order
        for k in [1, 2, 3]:
            kmer_freqs = features[f'kmer_{k}']
            vector.extend([kmer_freqs[kmer] for kmer in sorted(kmer_freqs.keys())])
            
        return np.array(vector)

    def bulk_process_sequences(self, sequences: List[str]) -> pd.DataFrame:
        """
        Process multiple sequences and return features dataframe.
        
        Args:
            sequences (List[str]): List of DNA sequences
            
        Returns:
            pd.DataFrame: DataFrame with all features
        """
        all_features = []
        
        for seq in sequences:
            features = self.generate_all_features(seq)
            flat_features = {}
            
            # Flatten physical properties
            for key, value in features['physical'].items():
                flat_features[f'physical_{key}'] = value
                
            # Flatten k-mer frequencies
            for k in [1, 2, 3]:
                for kmer, freq in features[f'kmer_{k}'].items():
                    flat_features[f'kmer{k}_{kmer}'] = freq
                    
            all_features.append(flat_features)
            
        return pd.DataFrame(all_features)