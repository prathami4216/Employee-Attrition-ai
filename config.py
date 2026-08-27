"""
=========================================================
Employee Attrition Prediction System
Configuration File
=========================================================
"""

import os

# =========================================================
# Project Root
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# Dataset Folder
# =========================================================

DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")

CSV_FILE = os.path.join(
    DATASET_FOLDER,
    "employee.csv"
)

XLSX_FILE = os.path.join(
    DATASET_FOLDER,
    "employee.xlsx"
)

# =========================================================
# Models Folder
# =========================================================

MODEL_FOLDER = os.path.join(BASE_DIR, "models")

PIPELINE_FILE = os.path.join(
    MODEL_FOLDER,
    "pipeline.pkl"
)

ENCODER_FILE = os.path.join(
    MODEL_FOLDER,
    "encoder.pkl"
)

# =========================================================
# Reports Folder
# =========================================================

REPORT_FOLDER = os.path.join(BASE_DIR, "reports")

CONFUSION_MATRIX = os.path.join(
    REPORT_FOLDER,
    "confusion_matrix.png"
)

FEATURE_IMPORTANCE = os.path.join(
    REPORT_FOLDER,
    "feature_importance.png"
)

DECISION_TREE = os.path.join(
    REPORT_FOLDER,
    "decision_tree.png"
)

METRICS_FILE = os.path.join(
    REPORT_FOLDER,
    "metrics.csv"
)

# =========================================================
# Assets Folder
# =========================================================

ASSET_FOLDER = os.path.join(BASE_DIR, "assets")

LOGO = os.path.join(
    ASSET_FOLDER,
    "logologo.png"
)

# =========================================================
# Target Column
# =========================================================

TARGET_COLUMN = "Attrition"

# =========================================================
# Columns To Remove
# =========================================================

DROP_COLUMNS = [

    "Employee_ID"

]

# =========================================================
# Random Seed
# =========================================================

RANDOM_STATE = 42

# =========================================================
# Train Test Split
# =========================================================

TEST_SIZE = 0.20

# =========================================================
# Decision Tree Parameters
# =========================================================

CRITERION = "entropy"

MAX_DEPTH = None

MIN_SAMPLES_SPLIT = 5

MIN_SAMPLES_LEAF = 2

# =========================================================
# Grid Search Parameters
# =========================================================

GRID_PARAMETERS = {

    "classifier__n_estimators": [100, 200, 300, 500],

    "classifier__max_depth": [4, 6, 8, 10, 12, None],

    "classifier__min_samples_split": [2, 5, 10],

    "classifier__min_samples_leaf": [1, 2, 4],

    "classifier__max_features": ["sqrt", "log2"],

    "classifier__class_weight": ["balanced", None]

}
# =========================================================
# Streamlit Page
# =========================================================

APP_TITLE = "Employee Attrition Prediction System"

APP_ICON = "📊"

LAYOUT = "wide"

# =========================================================
# Risk Thresholds
# =========================================================

HIGH_RISK = 0.75

MEDIUM_RISK = 0.50

LOW_RISK = 0.25

# =========================================================
# Supported Files
# =========================================================

SUPPORTED_FILES = [

    ".csv",

    ".xlsx"

]

# =========================================================
# Ensure Required Folders Exist
# =========================================================

os.makedirs(DATASET_FOLDER, exist_ok=True)

os.makedirs(MODEL_FOLDER, exist_ok=True)

os.makedirs(REPORT_FOLDER, exist_ok=True)

os.makedirs(ASSET_FOLDER, exist_ok=True)