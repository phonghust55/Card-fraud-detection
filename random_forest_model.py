import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
from utils import evaluate_model # Import the evaluation function

warnings.filterwarnings('ignore')

# Set random state for reproducibility
RANDOM_STATE = 42

# Load the data
df = pd.read_csv('creditcard.csv')

# Separate features and target
X = df.drop(columns='Class')
y = df['Class']

# Split into train and test sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=RANDOM_STATE, 
    stratify=y
)

print(f"Training set shape: {X_train.shape}, Target distribution: {np.bincount(y_train) if len(y_train) > 0 else 'empty'}")
print(f"Testing set shape: {X_test.shape}, Target distribution: {np.bincount(y_test) if len(y_test) > 0 else 'empty'}")

# Define model pipeline creation function
def create_random_forest_pipeline(sampling_method=None):
    steps = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]
    
    if sampling_method == 'smote':
        steps.append(('sampling', SMOTE(random_state=RANDOM_STATE)))
    elif sampling_method == 'undersampling':
        steps.append(('sampling', RandomUnderSampler(random_state=RANDOM_STATE)))
        
    steps.append(('model', RandomForestClassifier(random_state=RANDOM_STATE)))
    return ImbPipeline(steps)

if __name__ == "__main__":
    print("\nRunning Random Forest Model (with SMOTE)...\n")
    
    # Create and train the Random Forest model with SMOTE
    rf_pipeline_smote = create_random_forest_pipeline(sampling_method='smote')
    
    print("Training Random Forest (SMOTE)...")
    rf_pipeline_smote.fit(X_train, y_train)
    print("Training complete.")
    
    print("\nEvaluating Random Forest (SMOTE)...")
    evaluate_model(rf_pipeline_smote, X_test, y_test, model_name="Random_Forest_SMOTE_Standalone")
    print("\nRandom Forest (SMOTE) evaluation complete.")