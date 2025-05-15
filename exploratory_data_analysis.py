import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set random state for reproducibility
RANDOM_STATE = 42

# Load the data
df = pd.read_csv('creditcard.csv')

# 2. EXPLORATORY DATA ANALYSIS (EDA)
print("\n" + "=" * 50)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# Create a directory for plots if it doesn't exist
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created directory: {plots_dir}")
else:
    print(f"Directory already exists or no need to create: {plots_dir}")

# Distribution of each feature
def plot_feature_distributions(df_to_plot, save_dir='plots', random_state=42):
    plt.figure(figsize=(10, 6))
    plt.title('Transaction Amount Distribution')
    sns.histplot(data=df_to_plot, x='Amount', hue='Class', bins=50, kde=True)
    plt.savefig(os.path.join(save_dir, 'amount_distribution.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.title('Transaction Time Distribution')
    sns.histplot(data=df_to_plot, x='Time', hue='Class', bins=50, kde=True)
    plt.savefig(os.path.join(save_dir, 'time_distribution.png'))
    plt.close()
    
    fraud = df_to_plot[df_to_plot['Class'] == 1]
    # Ensure there are non-fraud samples before sampling
    if not df_to_plot[df_to_plot['Class'] == 0].empty and not fraud.empty:
        non_fraud_samples = min(len(fraud) * 5, len(df_to_plot[df_to_plot['Class'] == 0]))
        non_fraud = df_to_plot[df_to_plot['Class'] == 0].sample(n=non_fraud_samples, random_state=random_state)
        
        for feature in [col for col in df_to_plot.columns if col.startswith('V')]:
            plt.figure(figsize=(10, 6))
            plt.title(f'{feature} Distribution by Class')
            sns.kdeplot(data=fraud, x=feature, label='Fraud', fill=True) # Changed shade to fill
            sns.kdeplot(data=non_fraud, x=feature, label='Non-Fraud', fill=True) # Changed shade to fill
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'{feature}_distribution.png'))
            plt.close()
    else:
        print("Skipping V features distribution plots due to lack of fraud or non-fraud samples.")

print("Generating distribution plots...")
plot_feature_distributions(df, save_dir=plots_dir, random_state=RANDOM_STATE)
print(f"Distribution plots saved to '{plots_dir}' directory")

# Check for skewness in the data
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
skewness = df[numeric_features].skew()
high_skew_features = skewness[abs(skewness) > 1].sort_values(ascending=False)

print("\nFeatures with high skewness (|skew| > 1):")
if not high_skew_features.empty:
    print(high_skew_features)
else:
    print("No features with high skewness found.")

# Correlation analysis
# Ensure there are enough numeric features to calculate correlation
if len(numeric_features) > 1:
    correlation_matrix = df[numeric_features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
    plt.title('Correlation Matrix of Features')
    plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'))
    plt.close()
    print(f"Correlation matrix plot saved to '{os.path.join(plots_dir, 'correlation_matrix.png')}'")

    # Most correlated features with the target
    if 'Class' in correlation_matrix.columns:
        target_correlations = correlation_matrix['Class'].sort_values(ascending=False)
        print("\nTop features correlated with fraud (including Class itself):")
        print(target_correlations.head(11) if len(target_correlations) > 10 else target_correlations)
    else:
        print("\n'Class' column not in numeric features for correlation analysis.")
else:
    print("\nNot enough numeric features to compute correlation matrix.") 