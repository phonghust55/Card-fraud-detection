import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer # PowerTransformer was in original, kept for completeness
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import os
import joblib
from utils import evaluate_model # Import the evaluation function
from neural_network import create_neural_network_pipeline
from xgboost_model import create_xgboost_pipeline

warnings.filterwarnings('ignore')

# Set random state for reproducibility
RANDOM_STATE = 42

# Ensure the 'plots' directory exists (evaluate_model in utils also does this, but good practice here too)
if not os.path.exists('plots'):
    os.makedirs('plots')

# 3. PREPROCESSING AND TRAIN/TEST SPLIT (Combined from original Step 1 and 3)
print("\n" + "=" * 50)
print("STEP 1 & 2: DATA LOADING & PREPROCESSING (before model-specific sampling)")
print("=" * 50)

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

# 4. MODEL BUILDING AND EVALUATION
print("\n" + "=" * 50)
print("STEP 3: MODEL BUILDING AND EVALUATION (Multiple Models & Sampling)")
print("=" * 50)

# Define model pipeline creation functions (copied from original script)
def create_decision_tree_pipeline(sampling_method=None):
    steps = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]
    if sampling_method == 'smote':
        steps.append(('sampling', SMOTE(random_state=RANDOM_STATE)))
    elif sampling_method == 'undersampling':
        steps.append(('sampling', RandomUnderSampler(random_state=RANDOM_STATE)))
    steps.append(('model', DecisionTreeClassifier(random_state=RANDOM_STATE)))
    return ImbPipeline(steps)

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

def create_logistic_regression_pipeline(sampling_method=None):
    steps = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]
    if sampling_method == 'smote':
        steps.append(('sampling', SMOTE(random_state=RANDOM_STATE)))
    elif sampling_method == 'undersampling':
        steps.append(('sampling', RandomUnderSampler(random_state=RANDOM_STATE)))
    steps.append(('model', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)))
    return ImbPipeline(steps)

# Models to train and evaluate
models_to_run = {
    'Decision Tree (No Sampling)': create_decision_tree_pipeline(),
    'Decision Tree (SMOTE)': create_decision_tree_pipeline('smote'),
    'Decision Tree (Undersampling)': create_decision_tree_pipeline('undersampling'),
    'Random Forest (SMOTE)': create_random_forest_pipeline('smote'),
    'Logistic Regression (SMOTE)': create_logistic_regression_pipeline('smote'),
    'Neural Network (SMOTE)': create_neural_network_pipeline('smote'),
    'XGBoost (SMOTE)': create_xgboost_pipeline('smote')
}

results = {}

for name, model_pipeline in models_to_run.items():
    print(f"\nTraining {name}...")
    model_pipeline.fit(X_train, y_train)
    print(f"Training {name} complete.")
    
    print(f"Evaluating {name}...")
    # Replace spaces and parentheses for valid file names
    safe_model_name = name.replace(' ', '_').replace('(', '').replace(')', '')
    results[name] = evaluate_model(model_pipeline, X_test, y_test, safe_model_name)
    print(f"Evaluation of {name} complete.")

# 5. HYPERPARAMETER TUNING
print("\n" + "=" * 50)
print("STEP 4: HYPERPARAMETER TUNING")
print("=" * 50)

# Find the best model based on recall (as we want to catch as many frauds as possible)
if results: # ensure results is not empty
    best_model_name = max(results, key=lambda x: results[x]['recall'] if results[x]['recall'] is not None else -1)
    print(f"\nBest performing model (based on Recall from initial runs): {best_model_name}")

    tuned_model_pipeline = None
    param_grid = {}

    if 'Decision Tree' in best_model_name:
        tuned_model_pipeline = create_decision_tree_pipeline('smote' if 'SMOTE' in best_model_name else ('undersampling' if 'Undersampling' in best_model_name else None))
        param_grid = {
            'model__max_depth': [4, 6, 8, 10],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
    elif 'Random Forest' in best_model_name:
        tuned_model_pipeline = create_random_forest_pipeline('smote' if 'SMOTE' in best_model_name else ('undersampling' if 'Undersampling' in best_model_name else None))
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5, 10]
        }
    elif 'Logistic Regression' in best_model_name:
        tuned_model_pipeline = create_logistic_regression_pipeline('smote' if 'SMOTE' in best_model_name else ('undersampling' if 'Undersampling' in best_model_name else None))
        param_grid = {
            'model__C': [0.01, 0.1, 1.0, 10.0],
            'model__penalty': ['l1', 'l2'], # Removed 'elasticnet' as it requires l1_ratio
            'model__solver': ['liblinear', 'saga'] # Saga supports l1 and l2, liblinear supports l1, l2
        }
        # Filter solver based on penalty for Logistic Regression
        # param_grid = [
        #     {'model__penalty': ['l1'], 'model__solver': ['liblinear', 'saga'], 'model__C': [0.01, 0.1, 1, 10]},
        #     {'model__penalty': ['l2'], 'model__solver': ['liblinear', 'saga'], 'model__C': [0.01, 0.1, 1, 10]},
        # ]
    elif 'Neural Network' in best_model_name:
        tuned_model_pipeline = create_neural_network_pipeline('smote' if 'SMOTE' in best_model_name else None)
        param_grid = {
            'model__epochs': [5, 10, 15],
            'model__batch_size': [128, 256, 512],
            'model__model__units_layer1': [32, 64, 128],
            'model__model__dropout_rate': [0.2, 0.3, 0.4]
        }
    elif 'XGBoost' in best_model_name:
        tuned_model_pipeline = create_xgboost_pipeline('smote' if 'SMOTE' in best_model_name else None)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 4, 5],
            'model__learning_rate': [0.01, 0.1, 0.3],
            'model__min_child_weight': [1, 3, 5]
        }
    else:
        print("Could not determine the best model type for hyperparameter tuning or no models were run.")
        tuned_model_pipeline = None

    if tuned_model_pipeline and param_grid:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE) # Reduced splits for speed
        
        print(f"\nPerforming hyperparameter tuning for: {best_model_name}")
        print(f"Parameter grid: {param_grid}")
        print("This may take some time...")

        grid_search = GridSearchCV(
            estimator=tuned_model_pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='recall', # Prioritize recall for fraud detection
            n_jobs=-1,
            verbose=1 # Added verbosity
        )
        grid_search.fit(X_train, y_train)

        print("\nBest parameters found:")
        print(grid_search.best_params_)

        print("\nEvaluating tuned model...")
        tuned_model = grid_search.best_estimator_
        tuned_results = evaluate_model(tuned_model, X_test, y_test, "Tuned_" + best_model_name.replace(' ', '_').replace('(','').replace(')',''))
        results["Tuned_" + best_model_name] = tuned_results
        
        # Save the final tuned model
        final_model_filename = 'final_fraud_detection_model.joblib'
        joblib.dump(tuned_model, final_model_filename)
        print(f"\nFinal tuned model saved as '{final_model_filename}'")

    else:
        print("Skipping hyperparameter tuning as no base model or param_grid could be set up.")
elif not results:
    print("No models were evaluated, skipping hyperparameter tuning.")

# 6. FINAL COMPARISON
print("\n" + "=" * 50)
print("STEP 5: FINAL MODEL COMPARISON")
print("=" * 50)

if results: # Check if results dictionary is populated
    # Create a DataFrame for comparison
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics.get('accuracy'),
            'Precision (Fraud)': metrics.get('precision'),
            'Recall (Fraud)': metrics.get('recall'),
            'F1-Score (Fraud)': metrics.get('f1'),
            'AUC-ROC': metrics.get('auc'),
            'Avg Precision': metrics.get('avg_precision')
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string())
else:
    print("No results to compare.")

print("\n" + "=" * 50)
print("PIPELINE EXECUTION COMPLETE")
print("=" * 50) 