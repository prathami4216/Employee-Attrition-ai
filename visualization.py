"""
=========================================================
Employee Attrition Prediction System
Visualization Module
=========================================================
Creates:
1. Confusion Matrix
2. Feature Importance
3. Decision Tree
4. Class Distribution
5. Correlation Heatmap
=========================================================
"""

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import plot_tree
from sklearn.metrics import ConfusionMatrixDisplay


# ==========================================================
# Plot Confusion Matrix
# ==========================================================

def plot_confusion_matrix(cm, save_path=None):

    plt.figure(figsize=(6, 5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No", "Yes"]
    )

    disp.plot(values_format="d")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.close()


# ==========================================================
# Feature Importance
# ==========================================================

def plot_feature_importance(

    classifier,

    feature_names,

    save_path=None

):

    importance = classifier.feature_importances_

    data = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance

    })

    data = data.sort_values(

        by="Importance",

        ascending=False

    )

    top_features = data.head(15)

    colors = plt.cm.viridis(

        (top_features["Importance"] / top_features["Importance"].max()).values

    )

    plt.figure(figsize=(12,7))

    plt.barh(

        top_features["Feature"],

        top_features["Importance"],

        color=colors

    )

    plt.xlabel("Importance")

    plt.ylabel("Features")

    plt.title("Top 15 Feature Importances")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    if save_path:

        plt.savefig(

            save_path,

            dpi=300,

            bbox_inches="tight"

        )

    plt.close()

# ==========================================================
# Decision Tree
# ==========================================================
def plot_decision_tree(

    classifier,

    feature_names,

    save_path=None,

    display_depth=3

):

    """
    display_depth limits how many levels are DRAWN, independent of how
    deep the trained tree actually is. This keeps the figure readable
    even if the real tree has 15+ levels.
    """

    fig, ax = plt.subplots(figsize=(28, 16))

    plot_tree(

        classifier,

        feature_names=feature_names,

        class_names=["No","Yes"],

        filled=True,

        rounded=True,

        fontsize=10,

        impurity=True,

        proportion=True,

        max_depth=display_depth,

        ax=ax

    )

    ax.set_title(
        f"Decision Tree (showing top {display_depth} levels)",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout()

    if save_path:

        plt.savefig(

            save_path,

            dpi=200,

            bbox_inches="tight",

            facecolor="white"

        )

    plt.close()
# ==========================================================
# Attrition Distribution
# ==========================================================

def plot_attrition_distribution(

    dataframe,

    save_path=None

):

    counts = dataframe["Attrition"].value_counts()

    plt.figure(figsize=(6,6))

    plt.pie(

        counts,

        labels=counts.index,

        autopct="%1.1f%%",

        startangle=90

    )

    plt.title(

        "Employee Attrition Distribution"

    )

    if save_path:

        plt.savefig(

            save_path,

            dpi=300,

            bbox_inches="tight"

        )

    plt.close()


# ==========================================================
# Correlation Heatmap
# ==========================================================

def plot_correlation_heatmap(

    dataframe,

    save_path=None

):

    numeric = dataframe.select_dtypes(

        include="number"

    )

    correlation = numeric.corr()

    plt.figure(figsize=(10,8))

    plt.imshow(

        correlation,

        aspect="auto"

    )

    plt.colorbar()

    plt.xticks(

        range(len(correlation.columns)),

        correlation.columns,

        rotation=90

    )

    plt.yticks(

        range(len(correlation.columns)),

        correlation.columns

    )

    plt.title(

        "Correlation Heatmap"

    )

    plt.tight_layout()

    if save_path:

        plt.savefig(

            save_path,

            dpi=300,

            bbox_inches="tight"

        )

    plt.close()


# ==========================================================
# Model Summary
# ==========================================================

def print_model_summary(

    accuracy,

    precision,

    recall,

    f1

):

    print("\n")

    print("="*55)

    print("MODEL PERFORMANCE")

    print("="*55)

    print(f"Accuracy  : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"F1 Score  : {f1:.4f}")

    print("="*55)