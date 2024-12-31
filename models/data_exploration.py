"""
Enhanced data exploration module with advanced sequence analysis,
ML model evaluation, CUDA support, enhanced logging, and motif significance testing.
"""

import os
import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, List, Tuple, Optional, Any

# Biopython
from Bio import motifs
from Bio.Seq import Seq

# SciPy and scikit-learn
from scipy import stats
from scipy.stats import binom_test
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc
)

# We'll conditionally import PCA/t-SNE from scikit-learn or cuML
try:
    from cuml.decomposition import PCA as cumlPCA
    from cuml.manifold import TSNE as cumlTSNE
    HAS_CUML = True
except ImportError:
    from sklearn.decomposition import PCA as skPCA
    from sklearn.manifold import TSNE as skTSNE
    HAS_CUML = False

import logomaker


class SequenceExplorer:
    def __init__(
        self,
        log_filename: Optional[str] = None,
        use_cuda: bool = False
    ):
        """
        Initialize the sequence explorer with optional CUDA support
        and enhanced logging (console + file).
        
        Args:
            log_filename (str): If provided, logs will be written to this file
                                in addition to console.
            use_cuda (bool): If True, use RAPIDS cuML for PCA/t-SNE if available.
        """

        self.use_cuda = use_cuda and HAS_CUML  # only True if user wants CUDA AND cuML is installed
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # capture all logs at DEBUG and above

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler (INFO level)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # File handler (DEBUG level) if filename is given
        if log_filename:
            fh = logging.FileHandler(log_filename)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        
        # A basic log to confirm environment
        if self.use_cuda:
            self.logger.info("CUDA support enabled. Using RAPIDS cuML for PCA/t-SNE (if invoked).")
        else:
            self.logger.info("Using standard scikit-learn for PCA/t-SNE or cuML not installed.")

        plt.style.use('seaborn')
    
    def sequence_complexity_analysis(self, sequences: List[str]) -> pd.DataFrame:
        """
        Analyze sequence complexity using various metrics:
          - Linguistic complexity (k-mer coverage up to k=5, if sequence length >= 5)
          - Shannon entropy
          - GC skew = (G - C) / (G + C)
          - Longest homopolymer run (max run of repeated base)
        
        Args:
            sequences (List[str]): List of DNA sequences.
            
        Returns:
            pd.DataFrame: Complexity metrics for each sequence.
        """
        complexity_metrics = []
        
        for seq in sequences:
            seq_len = len(seq)
            if seq_len == 0:
                self.logger.warning("Encountered an empty sequence.")
                complexity_metrics.append({
                    'sequence_length': 0,
                    'entropy': np.nan,
                    'gc_skew': np.nan,
                    'longest_homopolymer': np.nan
                })
                continue

            # Calculate linguistic complexity for k=1..5 or up to length of seq
            kmers = {}
            max_k = min(seq_len, 6)
            for k in range(1, max_k):
                observed = len(set(seq[i:i+k] for i in range(seq_len - k + 1)))
                possible = min(4**k, seq_len - k + 1)
                kmers[f'complexity_k{k}'] = observed / possible if possible > 0 else 0.0
            
            # Shannon Entropy
            base_freqs = [seq.count(base)/seq_len for base in 'ATGC']
            entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in base_freqs)
            
            # GC skew
            g_count = seq.count('G')
            c_count = seq.count('C')
            if (g_count + c_count) > 0:
                gc_skew = (g_count - c_count) / (g_count + c_count)
            else:
                gc_skew = 0
            
            # Longest homopolymer run
            # e.g., "AAACCG" -> longest run of A is 3, C is 2, G is 1
            longest_run = 1
            current_run = 1
            for i in range(1, seq_len):
                if seq[i] == seq[i-1]:
                    current_run += 1
                    longest_run = max(longest_run, current_run)
                else:
                    current_run = 1
            
            metrics = {
                'sequence_length': seq_len,
                'entropy': entropy,
                'gc_skew': gc_skew,
                'longest_homopolymer': longest_run,
                **kmers
            }
            complexity_metrics.append(metrics)
        
        return pd.DataFrame(complexity_metrics)

    def motif_discovery(
        self,
        sequences: List[str],
        motif_length: int = 6,
        top_n: int = 5,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Discover recurring motifs in sequences, and test for statistical significance
        using a binomial test.
        
        Args:
            sequences (List[str]): List of DNA sequences
            motif_length (int): Length of motifs to search for
            top_n (int): Number of top motifs to return
            significance_level (float): p-value threshold for significance
            
        Returns:
            Dict containing:
              - 'top_motifs': list of (motif, count) for the most common motifs
              - 'motif_pvals': dict of motif -> p-value
              - 'significant_motifs': list of motifs passing significance threshold
              - 'pssm': PSSM matrix (str formatted)
              - 'motif_locations': dict of motif -> list of lists of positions in each seq
        """
        self.logger.info(f"Running motif discovery for {len(sequences)} sequences.")
        if not sequences:
            return {}
        
        # Convert sequences to Biopython Seq objects
        seq_records = [Seq(seq) for seq in sequences if seq]
        
        # If there's at least one sequence, create a motif object.
        # NOTE: This step is a bit simplistic for actual motif "discovery",
        # but we use Bio.motifs.create as a demonstration.
        m = motifs.create(seq_records)
        
        # Biopython can give us a PSSM
        pssm = m.pssm
        
        # Count motifs of length 'motif_length'
        motif_counts = {}
        total_positions = 0
        for seq in sequences:
            if len(seq) < motif_length:
                continue
            total_positions += (len(seq) - motif_length + 1)
            for i in range(len(seq) - motif_length + 1):
                sub = seq[i:i+motif_length]
                motif_counts[sub] = motif_counts.get(sub, 0) + 1
        
        # Get top motifs
        sorted_counts = sorted(motif_counts.items(), key=lambda x: x[1], reverse=True)
        top_motifs = sorted_counts[:top_n]
        
        # Binomial test for significance of each motif
        # H0: The motif occurs with probability p = (1/4)^motif_length (random)
        # x = number of occurrences, n = total positions, p = 1/4^motif_length
        p = (1/4) ** motif_length
        motif_pvals = {}
        for motif_seq, count in top_motifs:
            pval = binom_test(count, total_positions, p, alternative='greater')
            motif_pvals[motif_seq] = pval
        
        # Filter significant motifs based on significance_level
        significant_motifs = [
            motif_seq for motif_seq, pval in motif_pvals.items() if pval < significance_level
        ]
        
        # Plot a sequence logo from PSSM
        plt.figure(figsize=(10, 3))
        logomaker.Logo(pssm)
        plt.title('Sequence Logo (Biopython Motif)')
        
        # Gather motif locations in each sequence
        motif_locations = {
            motif_seq: self._find_motif_positions(sequences, motif_seq)
            for motif_seq, _ in top_motifs
        }
        
        return {
            'top_motifs': top_motifs,
            'motif_pvals': motif_pvals,
            'significant_motifs': significant_motifs,
            'pssm': pssm.format(),
            'motif_locations': motif_locations
        }
    
    def _find_motif_positions(self, sequences: List[str], motif: str) -> List[List[int]]:
        """Helper function to find motif positions in sequences."""
        positions = []
        for seq in sequences:
            pos = []
            for i in range(len(seq) - len(motif) + 1):
                if seq[i:i+len(motif)] == motif:
                    pos.append(i)
            positions.append(pos)
        return positions

    def evaluate_model_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation including various metrics and visualizations.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_prob: Prediction probabilities (for classification)
            
        Returns:
            Dict containing evaluation metrics and paths to generated plots
        """
        self.logger.info("Evaluating model performance.")
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = float(np.mean((y_true - y_pred) ** 2))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['mae'] = float(np.mean(np.abs(y_true - y_pred)))
        metrics['r2'] = float(1 - (np.sum((y_true - y_pred) ** 2) /
                                  np.sum((y_true - np.mean(y_true)) ** 2)))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Scatter plot
        axes[0,0].scatter(y_true, y_pred, alpha=0.5)
        axes[0,0].plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            'r--', lw=2
        )
        axes[0,0].set_title('Predicted vs Actual Values')
        axes[0,0].set_xlabel('Actual Values')
        axes[0,0].set_ylabel('Predicted Values')
        
        # Residual plot
        residuals = y_true - y_pred
        axes[0,1].scatter(y_pred, residuals, alpha=0.5)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_title('Residual Plot')
        axes[0,1].set_xlabel('Predicted Values')
        axes[0,1].set_ylabel('Residuals')
        
        # Residual histogram
        sns.histplot(residuals, ax=axes[1,0])
        axes[1,0].set_title('Residual Distribution')
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1,1])
        axes[1,1].set_title('Q-Q Plot')
        
        # If probabilities are provided (for classification)
        if y_prob is not None:
            plt.figure(figsize=(10, 5))
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall, precision)
            plt.subplot(1, 2, 2)
            plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            
            metrics['roc_auc'] = float(roc_auc)
            metrics['pr_auc'] = float(pr_auc)
        
        return metrics

    def analyze_model_robustness(
        self,
        model,
        sequences: List[str],
        n_perturbations: int = 100
    ) -> pd.DataFrame:
        """
        Analyze model robustness to sequence perturbations.
        
        Args:
            model: Trained model
            sequences: List of sequences to test
            n_perturbations: Number of random single-base perturbations per sequence
            
        Returns:
            pd.DataFrame containing robustness metrics for each sequence
        """
        self.logger.info(f"Analyzing model robustness for {len(sequences)} sequences.")
        results = []
        bases = list('ATGC')
        
        for seq in sequences:
            original_pred = model.predict([seq])[0]
            
            # Random single-base mutations
            perturbed_preds = []
            for _ in range(n_perturbations):
                if len(seq) == 0:
                    perturbed_preds.append(original_pred)
                    continue
                pos = np.random.randint(0, len(seq))
                new_base = np.random.choice([b for b in bases if b != seq[pos]])
                perturbed_seq = seq[:pos] + new_base + seq[pos+1:]
                pred = model.predict([perturbed_seq])[0]
                perturbed_preds.append(pred)
            
            results.append({
                'sequence': seq,
                'original_pred': float(original_pred),
                'mean_perturbed': float(np.mean(perturbed_preds)),
                'std_perturbed': float(np.std(perturbed_preds)),
                'max_deviation': float(max(abs(p - original_pred) for p in perturbed_preds))
            })
        
        return pd.DataFrame(results)

    def dimensionality_reduction(
        self,
        features: np.ndarray,
        method: str = 'pca',
        n_components: int = 2
    ) -> np.ndarray:
        """
        Perform dimensionality reduction on feature data using either PCA or t-SNE,
        optionally with GPU acceleration (cuML) if available and requested.
        
        Args:
            features (np.ndarray): Feature matrix
            method (str): 'pca' or 'tsne'
            n_components (int): Number of components or dimensions to reduce to
        
        Returns:
            np.ndarray: Transformed 2D or nD data
        """
        self.logger.info(f"Performing {method.upper()} with n_components={n_components}, "
                         f"use_cuda={self.use_cuda}.")
        if method.lower() == 'pca':
            if self.use_cuda:
                self.logger.debug("Using cuML PCA.")
                reducer = cumlPCA(n_components=n_components)
            else:
                self.logger.debug("Using scikit-learn PCA.")
                reducer = skPCA(n_components=n_components)
            return reducer.fit_transform(features)
        
        elif method.lower() == 'tsne':
            if self.use_cuda:
                self.logger.debug("Using cuML TSNE.")
                reducer = cumlTSNE(n_components=n_components, verbose=True)
            else:
                self.logger.debug("Using scikit-learn TSNE.")
                reducer = skTSNE(n_components=n_components, verbose=1)
            return reducer.fit_transform(features)
        
        else:
            raise ValueError("Invalid dimensionality reduction method. Choose 'pca' or 'tsne'.")

    def save_analysis_report(self, analyses: Dict[str, Any], output_dir: str = 'analysis_results'):
        """Enhanced save function with HTML report generation."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save numerical results
        results_file = output_path / 'analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(analyses, f, indent=2)
        
        # Save figures
        for i, fig in enumerate(plt.get_fignums()):
            plt.figure(fig)
            plt.savefig(output_path / f'figure_{i}.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # Generate HTML report
        html_content = self._generate_html_report(analyses, output_path)
        with open(output_path / 'report.html', 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Analysis results saved to '{output_dir}'")

    def _generate_html_report(self, analyses: Dict[str, Any], output_path: Path) -> str:
        """Generate HTML report from analyses."""
        html = """
        <html>
        <head>
            <title>Sequence Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .section { margin: 20px 0; }
                .figure { margin: 10px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
        <h1>Sequence Analysis Report</h1>
        """
        
        for analysis_name, results in analyses.items():
            html += f"<div class='section'><h2>{analysis_name}</h2>"
            
            if isinstance(results, dict):
                html += "<table><tr><th>Metric</th><th>Value</th></tr>"
                for metric, value in results.items():
                    html += f"<tr><td>{metric}</td><td>{value}</td></tr>"
                html += "</table>"
            
            # Attempt to load figure for this analysis (if we named it consistently)
            figure_file = f"figure_{analysis_name}.png"
            possible_path = output_path / figure_file
            if possible_path.exists():
                html += f"<div class='figure'><img src='{figure_file}' style='max-width:700px;'></div>"
            
            html += "</div>"
        
        html += "</body></html>"
        return html


