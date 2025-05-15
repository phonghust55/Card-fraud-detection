import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import numpy as np # Added for np.bincount if needed by other parts, though evaluate_model doesn't use it directly

# Ensure the 'plots' directory exists
if not os.path.exists('plots'):
    os.makedirs('plots')

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance with multiple metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Generate probabilities for ROC AUC
    auc_score = None
    avg_precision = None
    
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_prob)
            avg_precision = average_precision_score(y_test, y_prob)
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.savefig(f'plots/{model_name}_roc_curve.png')
            plt.close()
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.4f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend(loc="lower left")
            plt.savefig(f'plots/{model_name}_pr_curve.png')
            plt.close()
            
        except Exception as e:
            print(f"Error calculating AUC-ROC/PR for {model_name}: {str(e)}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'plots/{model_name}_confusion_matrix.png')
    plt.close()
    
    # Calculate additional metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0) # Added zero_division
    
    print(f"\n=== {model_name} Performance Metrics ===")
    print(f"Accuracy:       {report['accuracy']:.4f}")
    
    # Check if '1' (fraud class) is in report, handle if not (e.g. if no frauds predicted or actual)
    if '1' in report:
        print(f"Precision (Fraud): {report['1']['precision']:.4f}")
        print(f"Recall (Fraud):    {report['1']['recall']:.4f}")
        print(f"F1-Score (Fraud):  {report['1']['f1-score']:.4f}")
        precision_val = report['1']['precision']
        recall_val = report['1']['recall']
        f1_val = report['1']['f1-score']
    else: # Handle cases where the positive class might not be present in predictions
        print("Precision (Fraud): N/A (No fraud class in report)")
        print("Recall (Fraud):    N/A (No fraud class in report)")
        print("F1-Score (Fraud):  N/A (No fraud class in report)")
        precision_val = 0.0
        recall_val = 0.0
        f1_val = 0.0

    if auc_score is not None:
        print(f"AUC-ROC:        {auc_score:.4f}")
    if avg_precision is not None:
        print(f"Avg Precision:  {avg_precision:.4f}")
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0)) # Added zero_division
    
    return {
        'accuracy': report['accuracy'],
        'precision': precision_val,
        'recall': recall_val,
        'f1': f1_val,
        'auc': auc_score,
        'avg_precision': avg_precision
    } 