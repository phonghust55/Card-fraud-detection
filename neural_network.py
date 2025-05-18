import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
from utils import evaluate_model

warnings.filterwarnings('ignore')

# Set random state for reproducibility
RANDOM_STATE = 42

def create_neural_network_pipeline(sampling_method=None):
    """
    Create a pipeline for Neural Network model with optional sampling method
    """
    steps = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]
    
    if sampling_method == 'smote':
        steps.append(('sampling', SMOTE(random_state=RANDOM_STATE)))
    elif sampling_method == 'undersampling':
        steps.append(('sampling', RandomUnderSampler(random_state=RANDOM_STATE)))
    
    # Create MLPClassifier (Neural Network)
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=256,
        learning_rate='adaptive',
        max_iter=100,
        random_state=RANDOM_STATE,
        verbose=True
    )
    
    steps.append(('model', model))
    
    return ImbPipeline(steps)

if __name__ == '__main__':
    print("\nRunning Neural Network Model (with SMOTE)...\n")
    
    # Load data
    df = pd.read_csv('creditcard.csv')
    
    # Separate features and target
    X = df.drop(columns='Class')
    y = df['Class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    
    # Create and train the model with SMOTE
    print("Training Neural Network (SMOTE)...")
    model_pipeline = create_neural_network_pipeline('smote')
    model_pipeline.fit(X_train, y_train)
    print("Training complete.")
    
    # Evaluate the model
    print("\nEvaluating Neural Network (SMOTE)...")
    evaluate_model(model_pipeline, X_test, y_test, "Neural_Network_SMOTE_Standalone")
    print("\nNeural Network (SMOTE) evaluation complete.")
