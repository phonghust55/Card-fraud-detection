import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set random state for reproducibility (though not directly used in this script)
RANDOM_STATE = 42

# 1. DATA UNDERSTANDING
print("=" * 50)
print("STEP 1: DATA UNDERSTANDING")
print("=" * 50)

# Load the data
df = pd.read_csv('creditcard.csv')

# Display basic information
print("\nDataset Shape:", df.shape)
print("\nFeature Information:")
# df.info() can be verbose, print() wraps it for better console output control
import io
buffer = io.StringIO()
df.info(buf=buffer)
s_info = buffer.getvalue()
print(s_info)

print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found")

# Check class distribution
class_distribution = df['Class'].value_counts()
print("\nClass Distribution:")
print(class_distribution)
if 0 in class_distribution and 1 in class_distribution and class_distribution[0] > 0:
    print(f"Fraud ratio: {class_distribution[1] / class_distribution[0]:.6f}")
else:
    print("Could not calculate fraud ratio due to missing classes or zero count in non-fraud class.") 