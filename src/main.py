import pandas as pd
import numpy as np

def convert_to_dummy(df, feature, exclude_rank=0):
    # Create dummy variables
    dummies = pd.get_dummies(df[feature], prefix=feature)
    
    # Exclude the most frequent category
    most_frequent = df[feature].value_counts().index[exclude_rank]
    dummies.drop(f'{feature}_{most_frequent}', axis=1, inplace=True)
    
    # Drop the original column and add dummies
    df = df.drop(columns=[feature]).join(dummies)
    
    return df

def create_category_buckets(df, column, bins, labels, use_quantile=True):
    # Choose between quantile or equal-length binning
    if use_quantile:
        df[f'gp_{column}'] = pd.qcut(df[column], q=bins, labels=labels)  # Quantile bins
    else:
        df[f'gp_{column}'] = pd.cut(df[column], bins=bins, labels=labels)  # Equal-length bins
    
    return df

def calculate_woe_iv(data, target, feature):
    # Ensure the feature is treated as categorical
    data[feature] = data[feature].astype(str)  # Convert to string if not already
    
    # Calculate the total number of positive and negative cases
    total_good = (data[target] == 0).sum()
    total_bad = (data[target] == 1).sum()

    # Group by feature and calculate the WOE for each category
    grouped = data.groupby(feature).agg({target: ['count', 'sum']})
    grouped.columns = ['total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']
    
    # Avoid division by zero and calculate the proportion of good/bad
    grouped['bad_rate'] = grouped['bad'] / total_bad
    grouped['good_rate'] = grouped['good'] / total_good
    grouped['woe'] = np.log(grouped['good_rate'] / grouped['bad_rate']).replace([np.inf, -np.inf], 0)
    
    # Calculate IV for each category and sum it
    grouped['iv'] = (grouped['good_rate'] - grouped['bad_rate']) * grouped['woe']
    iv = grouped['iv'].sum()

    return grouped[['woe']], iv

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def evaluate_model_performance(model, X_test, y_test):
    """
    Function to evaluate and print performance metrics for a given model.
    
    Parameters:
    - model: Trained model to evaluate
    - X_test: Test feature data
    - y_test: Test target data
    
    Prints:
    - Accuracy, Precision, Recall, F1 Score, AUC Score
    - Confusion Matrix
    - Classification Report
    """
    # Predict labels
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    # Print metrics
    print(f"Model: {model.__class__.__name__}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
