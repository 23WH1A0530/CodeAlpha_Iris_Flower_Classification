# =============================================================================
# IRIS FLOWER CLASSIFICATION - CodeAlpha Internship Project
# Author   : CodeAlpha Intern
# Dataset  : Iris Dataset (sklearn built-in)
# Model    : Random Forest Classifier
# Task     : Multi-class Classification (3 species)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORT LIBRARIES
# ─────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                          # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. SETUP: Create output directory for plots
# ─────────────────────────────────────────────────────────────────────────────
IMAGES_DIR = "./images"

if not os.path.isdir(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)        # Create 'images/' folder if it doesn't exist

print("=" * 60)
print("    IRIS FLOWER CLASSIFICATION — CodeAlpha Internship")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/7] Loading Iris Dataset...")

iris = load_iris()                             # Load built-in Iris dataset from sklearn

# Convert to Pandas DataFrame for easy manipulation
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names                 # 4 feature columns
)
df['species'] = iris.target                    # Add numeric target column (0, 1, 2)
df['species_name'] = df['species'].map(        # Map numbers → species names
    {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
)

print(f"  ✔ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# 4. DISPLAY DATASET INFORMATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Dataset Information")
print("─" * 60)

# Basic shape
print(f"\n  Shape      : {df.shape}")
print(f"  Features   : {list(iris.feature_names)}")
print(f"  Target     : {list(iris.target_names)}")

# First 5 rows
print("\n  ── First 5 Rows ──")
print(df.head().to_string(index=False))

# Data types
print("\n  ── Data Types ──")
print(df.dtypes.to_string())

# Statistical Summary
print("\n  ── Statistical Summary ──")
print(df.describe().round(2).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 5. CHECK MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] Checking Missing Values...")
print("─" * 60)

missing = df.isnull().sum()
print(f"\n  Missing values per column:\n{missing.to_string()}")

if missing.sum() == 0:
    print("\n  ✔ No missing values found! Dataset is clean.")
else:
    print(f"\n  ⚠ Total missing values: {missing.sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Performing Exploratory Data Analysis (EDA)...")
print("─" * 60)

# Species distribution
print("\n  ── Species Distribution ──")
print(df['species_name'].value_counts().to_string())

# Mean feature values per species
print("\n  ── Mean Feature Values per Species ──")
print(df.groupby('species_name')[iris.feature_names].mean().round(2).to_string())

# ── PLOT 1: Species Count Plot ──────────────────────────────────────────────
print("\n  Generating: Species Count Plot...")

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#4C72B0', '#55A868', '#C44E52']     # Distinct colors for 3 species

df['species_name'].value_counts().plot(
    kind='bar',
    color=colors,
    ax=ax,
    edgecolor='black',
    linewidth=0.8
)

ax.set_title('Iris Species Count Distribution', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Species', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Add count labels on top of each bar
for bar in ax.patches:
    ax.annotate(
        f'{int(bar.get_height())}',
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 5),
        textcoords='offset points',
        ha='center',
        fontsize=12,
        fontweight='bold'
    )

ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
count_plot_path = os.path.join(IMAGES_DIR, "species_count_plot.png")
plt.savefig(count_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✔ Saved → {count_plot_path}")

# ── PLOT 2: Pair Plot ────────────────────────────────────────────────────────
print("  Generating: Pair Plot (this may take a moment)...")

pair_df = df[list(iris.feature_names) + ['species_name']].copy()

pair_plot = sns.pairplot(
    pair_df,
    hue='species_name',                        # Color by species
    palette={'setosa': '#4C72B0', 'versicolor': '#55A868', 'virginica': '#C44E52'},
    diag_kind='kde',                           # KDE on diagonal
    plot_kws={'alpha': 0.6, 's': 50},
    height=2.2
)
pair_plot.figure.suptitle(
    'Iris Dataset — Pair Plot of All Features',
    fontsize=15,
    fontweight='bold',
    y=1.02
)

pair_plot_path = os.path.join(IMAGES_DIR, "pair_plot.png")
pair_plot.savefig(pair_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✔ Saved → {pair_plot_path}")

# ── PLOT 3: Correlation Heatmap ──────────────────────────────────────────────
print("  Generating: Correlation Heatmap...")

fig, ax = plt.subplots(figsize=(8, 6))

# Compute correlation on numeric feature columns only
corr_matrix = df[iris.feature_names].corr()

sns.heatmap(
    corr_matrix,
    annot=True,                                # Show values in cells
    fmt='.2f',
    cmap='coolwarm',                           # Diverging color map
    center=0,
    linewidths=0.5,
    linecolor='white',
    square=True,
    ax=ax,
    annot_kws={'size': 12}
)

ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

plt.tight_layout()
heatmap_path = os.path.join(IMAGES_DIR, "correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✔ Saved → {heatmap_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. PREPARE DATA FOR TRAINING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] Preparing Data for Model Training...")
print("─" * 60)

X = df[iris.feature_names].values             # Feature matrix  (150 × 4)
y = df['species'].values                      # Target vector   (150,)

# Train/Test Split — 80% train, 20% test, stratified to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,                             # 20% test set
    random_state=42,                           # Reproducibility seed
    stratify=y                                 # Equal class distribution in both splits
)

print(f"\n  Total samples   : {len(X)}")
print(f"  Training set    : {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
print(f"  Test set        : {len(X_test)} samples  ({len(X_test)/len(X)*100:.0f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 8. TRAIN RANDOM FOREST CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/7] Training Random Forest Classifier...")
print("─" * 60)

# Initialize model
rf_model = RandomForestClassifier(
    n_estimators=100,                          # 100 decision trees in the forest
    max_depth=None,                            # Trees grow until all leaves are pure
    min_samples_split=2,                       # Min samples to split an internal node
    random_state=42,                           # Reproducibility
    n_jobs=-1                                  # Use all available CPU cores
)

rf_model.fit(X_train, y_train)                 # Train the model
print("  ✔ Model trained successfully!")

# Feature importances
print("\n  ── Feature Importances ──")
importances = rf_model.feature_importances_
for feat, imp in sorted(zip(iris.feature_names, importances), key=lambda x: -x[1]):
    bar = '█' * int(imp * 40)
    print(f"  {feat:<30} {imp:.4f}  {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. EVALUATE MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] Evaluating Model Performance...")
print("─" * 60)

y_pred = rf_model.predict(X_test)              # Predict on test set

# ── Accuracy ─────────────────────────────────────────────────────────────────
accuracy = accuracy_score(y_test, y_pred)
print(f"\n  ── Accuracy ──")
print(f"  Test Accuracy : {accuracy * 100:.2f}%")

# ── Classification Report ─────────────────────────────────────────────────────
print("\n  ── Classification Report ──")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# ── Confusion Matrix ──────────────────────────────────────────────────────────
print("  ── Confusion Matrix ──")
cm = confusion_matrix(y_test, y_pred)

# Print raw confusion matrix
cm_df = pd.DataFrame(
    cm,
    index=[f'Actual: {n}' for n in iris.target_names],
    columns=[f'Pred: {n}' for n in iris.target_names]
)
print(f"\n{cm_df.to_string()}\n")

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)
disp.plot(
    ax=ax,
    cmap='Blues',
    colorbar=True,
    xticks_rotation=15
)
ax.set_title(
    f'Confusion Matrix — Random Forest\n(Test Accuracy: {accuracy*100:.2f}%)',
    fontsize=14,
    fontweight='bold',
    pad=12
)
plt.tight_layout()
cm_path = os.path.join(IMAGES_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✔ Confusion matrix plot saved → {cm_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("    PROJECT COMPLETE — SUMMARY")
print("=" * 60)
print(f"\n  Model          : Random Forest Classifier")
print(f"  Trees          : 100 estimators")
print(f"  Train/Test     : 80% / 20% split")
print(f"  Test Accuracy  : {accuracy * 100:.2f}%")
print(f"\n  Plots saved to : ./{IMAGES_DIR}/")
print(f"    • species_count_plot.png")
print(f"    • pair_plot.png")
print(f"    • correlation_heatmap.png")
print(f"    • confusion_matrix.png")
print("\n  ✔ All tasks completed successfully!")
print("=" * 60)