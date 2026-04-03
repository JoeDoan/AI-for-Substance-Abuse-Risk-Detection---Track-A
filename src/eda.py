import os
import pandas as pd
from datasets import load_dataset
import sys

# Redirect output to make it easy to copy for reports
output_file = "data/eda_results.txt"
with open(output_file, 'w') as f:
    sys.stdout = f

    print("=== Exploratory Data Analysis (EDA) ===")
    print("\n--- 1. Mental Health Corpus (Local CSV) ---")
    try:
        df_local = pd.read_csv("data/mental_health.csv")
        print(f"Shape: {df_local.shape}")
        print("\nColumns:")
        print(df_local.dtypes)
        print("\nFirst 3 Rows:")
        print(df_local.head(3))
        
        # Determine the target column. It's often 'label' or 'class'
        target_col = None
        for col in ['label', 'class', 'target', 'status', 'sentiment', 'category']:
            if col in df_local.columns:
                target_col = col
                break
                
        if target_col:
            print(f"\nLabel Distribution for {target_col}:")
            print(df_local[target_col].value_counts(normalize=True).round(3))
        else:
            print(f"\nNo standard label column found. Unique values in all columns:")
            for c in df_local.columns:
                 if df_local[c].nunique() < 20: 
                     print(f"  {c}: {df_local[c].unique()}")
    except Exception as e:
        print(f"Error loading local CSV: {e}")

    print("\n--- 2. Reddit Mental Health Classification (HuggingFace) ---")
    try:
        ds = load_dataset("kamruzzaman-asif/reddit-mental-health-classification")
        print(f"Dataset Dictionary: {ds.keys()}")
        
        # Load train split into pandas
        if 'train' in ds:
            df_hf = ds['train'].to_pandas()
            print(f"\nShape of Train Split: {df_hf.shape}")
            print("\nColumns:")
            print(df_hf.dtypes)
            print("\nFirst 3 Rows:")
            print(df_hf.head(3))
            
            # Label distribution
            if 'label' in df_hf.columns:
                print("\nLabel Distribution:")
                print(df_hf['label'].value_counts(normalize=True).round(3))
            elif 'class' in df_hf.columns:
                print("\nLabel Distribution:")
                print(df_hf['class'].value_counts(normalize=True).round(3))
            else:
                 for c in df_hf.columns:
                     if df_hf[c].nunique() < 20: 
                         print(f"\nDistribution for {c}:")
                         print(df_hf[c].value_counts(normalize=True).round(3))
    except Exception as e:
        print(f"Error loading HF dataset: {e}")

sys.stdout = sys.__stdout__
print(f"EDA complete. Results saved to {output_file}")
