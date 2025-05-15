import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score

# Load data
print("Loading data...")
df = pd.read_csv('creditcard.csv')

# Separate features and target
X = df.drop(columns='Class')
y = df['Class']

# Split into train and test sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)
print("Data split complete.")

# Define a function to build pipelines
def create_pipeline(model):
    """
    Build an imbalanced-learn pipeline that:
      1) Imputes missing values with column means
      2) Standardizes features to zero mean & unit variance
      3) Applies SMOTE to balance classes
      4) Fits the specified model
    """
    return ImbPipeline([
        ('imputer', SimpleImputer(strategy='mean')),  
        ('scaler', StandardScaler()),                 
        ('smote', SMOTE(random_state=42)),            
        ('model', model)                              
    ])

# Instantiate pipelines for each model
pipeline_dt = create_pipeline(DecisionTreeClassifier(max_depth=6, random_state=42))
pipeline_rf = create_pipeline(RandomForestClassifier(n_estimators=100, random_state=42))
pipeline_lr = create_pipeline(LogisticRegression(random_state=42))

# Define models to evaluate
models = [
    ('Decision Tree', pipeline_dt),
    ('Random Forest', pipeline_rf),
    ('Logistic Regression', pipeline_lr)
]

# Store results for comparison
results = {}

# Train and evaluate each model
for name, pipeline in models:
    try:
        print(f"\n{'='*50}")
        print(f"EVALUATING: {name}")
        print(f"{'='*50}")
        
        # Train
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        print(f"Training {name} complete.")
        
        # Predict
        print(f"Generating predictions with {name}...")
        y_pred = pipeline.predict(X_test)
        print(f"Prediction with {name} complete.")
        
        # Generate probabilities for ROC AUC
        auc_score = None
        if hasattr(pipeline, "predict_proba"):
            try:
                print(f"Calculating AUC-ROC for {name}...")
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_prob)
                print(f"AUC-ROC calculation for {name} complete.")
            except Exception as e:
                print(f"Error calculating AUC-ROC for {name}: {str(e)}")
        else:
            print(f"{name} doesn't support predict_proba.")
        
        # Calculate additional metrics
        print(f"Calculating metrics for {name}...")
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"Metrics calculation for {name} complete.")
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_roc': auc_score
        }
        
        # Report
        print(f"\n=== {name} Performance Metrics ===")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        if auc_score is not None:
            print(f"AUC-ROC:   {auc_score:.4f}")
        else:
            print("AUC-ROC:   Not available")
        
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred, digits=4))
        
    except Exception as e:
        print(f"Error processing {name}: {str(e)}")
        import traceback
        traceback.print_exc()

# Print comparison of all models
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)
print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'AUC-ROC':<10}")
print("-"*70)

for name, metrics in results.items():
    print(f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['f1_score']:<10.4f}", end="")
    if metrics['auc_roc'] is not None:
        print(f" {metrics['auc_roc']:<10.4f}")
    else:
        print(" N/A        ")

print("="*70) 