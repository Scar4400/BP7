import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib
import plotly.graph_objects as go

def prepare_data_for_modeling(df):
    # Assuming 'result' is our target variable
    X = df.drop(["result", "timestamp", "team_name", "opponent_name"], axis=1)
    y = df["result"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=5, n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)

    return rf_grid.best_estimator_

def train_xgboost(X_train, y_train):
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb = XGBClassifier(random_state=42)
    xgb_grid = GridSearchCV(xgb, xgb_params, cv=5, n_jobs=-1, verbose=1)
    xgb_grid.fit(X_train, y_train)

    return xgb_grid.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_feature_importance(model, X):
    feature_importance = model.feature_importances_
    feature_names = X.columns

    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    fig = go.Figure(go.Bar(
        y=feature_names[sorted_idx],
        x=feature_importance[sorted_idx],
        orientation='h'
    ))

    fig.update_layout(
        title='Feature Importance',
        yaxis_title='Features',
        xaxis_title='Importance',
        height=800,
        width=800
    )

    fig.write_html("feature_importance.html")

def train_and_evaluate_models(df):
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(df)

    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)

    plot_feature_importance(rf_model, X_train)

    # Save the best model
    best_model = rf_model if rf_metrics['f1'] > xgb_metrics['f1'] else xgb_model
    joblib.dump(best_model, 'best_model.joblib')

    return {
        'random_forest': rf_metrics,
        'xgboost': xgb_metrics
    }

if __name__ == "__main__":
    # For testing purposes
    import json
    from data_preprocessing import preprocess_data
    from feature_engineering import create_features

    with open("sample_data.json", "r") as f:
        sample_data = json.load(f)

    processed_data = preprocess_data(sample_data)
    featured_data = create_features(processed_data)

    results = train_and_evaluate_models(featured_data)
    print("Model performance:")
    print(json.dumps(results, indent=2))

