# pipeline_test.py
import os
import sys
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Add the project root to Python path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.pipeline import SyntheticBioDataPipeline

def test_pipeline():
    # Initialize pipeline
    pipeline = SyntheticBioDataPipeline(email="your.email@example.com")
    
    try:
        # Fetch some promoter sequences
        print("Fetching sequences from NCBI...")
        sequences = pipeline.fetch_from_ncbi(
            query="promoter[Title] AND bacteria[Organism] AND 100:300[Sequence Length]",
            max_results=10
        )
        
        # Process sequences
        print("Processing sequences...")
        df = pipeline.process_sequences(sequences)
        
        # Save processed data
        print("Saving data...")
        pipeline.save_data(df, "test_promoters.csv")
        
        # Load and verify data
        print("Loading data...")
        loaded_df = pipeline.load_data("test_promoters.csv")
        print("\nLoaded data preview:")
        print(loaded_df.head())
        print("\nData shape:", loaded_df.shape)
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_pipeline()