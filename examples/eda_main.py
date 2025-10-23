"""
EDA Main Script for AKI Prediction Project

This script performs exploratory data analysis (EDA) using functions
from the utility module (utils.py). It loads the VitalDB dataset,
preprocesses it, visualizes key statistics, and inspects target distribution.
"""


import sys
import os
import json
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    setup_plotting,
    load_vitaldb_data,
    preprocess_data,
    prepare_train_test_data
)

def main():
    print("🚀 Starting EDA Pipeline...")
    setup_plotting()

    # === 1. Load raw dataset ===
    df = load_vitaldb_data()
    print("\n🧾 Raw Data Overview:")
    print(df.head())
    print("\n📋 Data Info:")
    print(df.info())
    print("\n📈 Missing values summary:")
    print(df.isnull().sum().sort_values(ascending=False).head(20))

    # === 2. Preprocess ===
    X, y, feature_names = preprocess_data(df)

    # === 3. Train/Test Split ===
    data_dict = prepare_train_test_data(X, y)
    X_train, X_test = data_dict["X_train_dict"]["imputed"], data_dict["X_test_dict"]["imputed"]
    y_train, y_test = data_dict["y_train"], data_dict["y_test"]

    # === 4. Visualize ===
    print("\n🎯 Target distribution:")
    sns.countplot(x=y)
    plt.title("Target Distribution (AKI = 1, Non-AKI = 0)")
    plt.xlabel("AKI")
    plt.ylabel("Count")
    plt.show()

    # Correlation heatmap
    df_proc = pd.DataFrame(X_train, columns=feature_names)
    corr = df_proc.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # === 5. Summary statistics ===
    print("\n📊 Basic statistics for numeric features:")
    print(df_proc.describe().T.head(10))

    print("\n✅ EDA completed successfully!")

if __name__ == "__main__":
    main()
