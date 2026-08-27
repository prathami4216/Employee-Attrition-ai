"""
===========================================================
Employee Attrition Prediction System
Model Training Script
===========================================================
Author : Prathami Sawant
Algorithm : Decision Tree Classifier
Criterion : Entropy (ID3 Style)
===========================================================
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report 
)

try:
    # When package is installed or run as a module
    from utils.preprocess import (
        prepare_training_data,
        create_preprocessor,
        save_pipeline,
        load_dataset,
        clean_dataset,
        dataset_summary
    )
except ImportError:
    # When running the script directly
    from utils.preprocess import (
        prepare_training_data,
        create_preprocessor,
        save_pipeline,
        load_dataset,
        clean_dataset,
        dataset_summary
    )

from config import *
# Display Heading
# ==========================================================

def heading():

    print("\n")
    print("="*65)
    print("EMPLOYEE ATTRITION PREDICTION SYSTEM")
    print("="*65)
    print("Decision Tree Classifier")
    print("="*65)

# ==========================================================
# Load Training Data
# ==========================================================

def load_training_data():

    print("\nLoading Dataset...")

    X, y, preprocessor = prepare_training_data()

    print("\nDataset Loaded Successfully")

    print("Samples :", len(X))

    print("Features :", len(X.columns))

    print("Target :", TARGET_COLUMN)

    return X, y, preprocessor

# ==========================================================
# Split Dataset
# ==========================================================

def split_dataset(X, y):

    print("\nSplitting Dataset...")

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )

    print("Training Samples :", len(X_train))

    print("Testing Samples  :", len(X_test))

    return X_train, X_test, y_train, y_test

# ==========================================================
# Create Pipeline
# ==========================================================

def build_pipeline(preprocessor):

    pipeline = ImbPipeline(

        [
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                criterion=CRITERION,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]

    )

    return pipeline
# ==========================================================
# Hyperparameter Tuning
# ==========================================================

def tune_model(pipeline, X_train, y_train):

    print("\nPerforming Randomized Search...")

    search = RandomizedSearchCV(

        estimator=pipeline,

        param_distributions=GRID_PARAMETERS,

        n_iter=60,

        cv=5,

        scoring="f1_weighted",

        random_state=RANDOM_STATE,

        n_jobs=-1

    )

    search.fit(

        X_train,

        y_train

    )

    print("\nBest Parameters")

    print(search.best_params_)

    print("Best CV F1 Score :", round(search.best_score_,4))

    return search.best_estimator_

# ==========================================================
# Train Model
# ==========================================================

def train_model(model, X_train, y_train):

    print("\nTraining Decision Tree...")

    model.fit(

        X_train,

        y_train

    )

    print("Training Completed.")

    return model

from visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_decision_tree
) 

import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report
    }

    metrics_json_path = os.path.join(MODEL_FOLDER, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nMetrics saved to:", metrics_json_path)

    return metrics, cm, predictions


if __name__ == "__main__":

    import os

    heading()

    X, y, preprocessor = load_training_data()
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    pipeline = build_pipeline(preprocessor)
    best_model = tune_model(pipeline, X_train, y_train)
    trained_model = train_model(best_model, X_train, y_train)

    metrics, cm, predictions = evaluate_model(trained_model, X_test, y_test)

    print_model_summary = None  # placeholder if you want to import from visualization
    print("\nAccuracy :", round(metrics["accuracy"], 4))
    print("F1 Score :", round(metrics["f1_score"], 4))

    plot_confusion_matrix(cm, save_path=CONFUSION_MATRIX)
    plot_feature_importance(
        trained_model.named_steps["classifier"],
        trained_model.named_steps["preprocessor"].get_feature_names_out(),
        save_path=FEATURE_IMPORTANCE
    )
    plot_decision_tree(
        trained_model.named_steps["classifier"].estimators_[0],
        trained_model.named_steps["preprocessor"].get_feature_names_out(),
        save_path=DECISION_TREE
    )

    save_pipeline(trained_model)

    print("\nTraining pipeline complete.")