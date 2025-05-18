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
    
    steps.append(('model', XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )))
    
    return ImbPipeline(steps)

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('creditcard.csv')
    
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
    model_pipeline = create_xgboost_pipeline('smote')
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model_pipeline, X_test, y_test, "XGBoost_SMOTE_Standalone")