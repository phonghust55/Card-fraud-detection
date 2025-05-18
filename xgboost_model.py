import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
from utils import evaluate_model

warnings.filterwarnings('ignore')

def create_xgboost_pipeline(sampling_method=None):
    """
    Create a pipeline for XGBoost model with optional sampling method
    """
    steps = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]
    
    if sampling_method == 'smote':
        steps.append(('sampling', SMOTE(random_state=42)))
    elif sampling_method == 'undersampling':
        steps.append(('sampling', RandomUnderSampler(random_state=42)))
    
    # Improved XGBoost configuration for fraud detection
    steps.append(('model', XGBClassifier(
        n_estimators=200,          # Increased from 100
        learning_rate=0.05,        # Decreased from 0.1
        max_depth=5,               # Increased from 4
        min_child_weight=3,        # Added parameter
        subsample=0.8,             # Added parameter
        colsample_bytree=0.8,      # Added parameter
        scale_pos_weight=50,       # Added to handle class imbalance
        random_state=42,
        use_label_encoder=False,   # Avoid warning
        eval_metric='auc'          # Better metric for imbalanced data
    )))
    
    return ImbPipeline(steps)

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('creditcard.csv')
    
    # Calculate class weight
    n_negative = len(df[df['Class'] == 0])
    n_positive = len(df[df['Class'] == 1])
    scale_pos_weight = n_negative/n_positive
    
    print(f"Data Statistics:")
    print(f"Number of normal transactions: {n_negative}")
    print(f"Number of fraud transactions: {n_positive}")
    print(f"Ratio (negative/positive): {scale_pos_weight:.2f}")
    
    # Separate features and target
    X = df.drop(columns='Class')
    y = df['Class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3, 
        random_state=42, 
        stratify=y
    )
    
    # Create and train the model with SMOTE
    print("\nTraining XGBoost model with improved parameters...")
    model_pipeline = create_xgboost_pipeline('smote')
    model_pipeline.fit(X_train, y_train)
    print("Training complete.")
    
    # Evaluate the model
    print("\nEvaluating XGBoost model...")
    evaluate_model(model_pipeline, X_test, y_test, "XGBoost_SMOTE_Standalone")