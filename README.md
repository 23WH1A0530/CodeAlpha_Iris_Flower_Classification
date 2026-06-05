# 🌸 Iris Flower Classification
### CodeAlpha Machine Learning Internship — Task 1

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

This project classifies **Iris flowers** into three species — *Setosa*, *Versicolor*, and *Virginica* — based on their petal and sepal measurements. It uses a **Random Forest Classifier** trained on the classic Iris dataset from `sklearn`.

This project was built as **Task 1** of the **CodeAlpha Machine Learning Internship**.

---

## 🌼 Dataset

| Property       | Details                                |
|----------------|----------------------------------------|
| Source         | `sklearn.datasets.load_iris()`         |
| Samples        | 150 (50 per class)                     |
| Features       | 4 (sepal length/width, petal length/width) |
| Target Classes | setosa, versicolor, virginica          |
| Missing Values | None                                   |

---

## 📁 Folder Structure

```
iris_flower_classification/
│
├── iris_classification.py   ← Main Python script
├── requirements.txt         ← Python dependencies
├── README.md                ← Project documentation
│
└── images/                  ← Auto-generated plots
    ├── species_count_plot.png
    ├── pair_plot.png
    ├── correlation_heatmap.png
    └── confusion_matrix.png
```

---

## 🔬 What the Script Does

### Step-by-Step Pipeline

| Step | Task |
|------|------|
| 1️⃣  | Load Iris dataset using `sklearn` |
| 2️⃣  | Convert to Pandas DataFrame |
| 3️⃣  | Display shape, types, and statistics |
| 4️⃣  | Check for missing values |
| 5️⃣  | Perform Exploratory Data Analysis (EDA) |
| 6️⃣  | Generate and save 3 EDA plots |
| 7️⃣  | Split data — 80% train / 20% test |
| 8️⃣  | Train Random Forest Classifier (100 trees) |
| 9️⃣  | Evaluate: Accuracy, Confusion Matrix, Classification Report |
| 🔟  | Save confusion matrix plot |

---

## 📊 Plots Generated

### 1. Species Count Plot
> Bar chart showing the distribution of the 3 Iris species (50 each).

### 2. Pair Plot
> Scatter matrix of all 4 features, colored by species. Reveals linear separability of *setosa*.

### 3. Correlation Heatmap
> Pearson correlation between the 4 numeric features. Petal length & width are highly correlated (≈0.96).

### 4. Confusion Matrix
> Model prediction results on the 20% test set, with accuracy displayed in the title.

---

## 🤖 Model: Random Forest Classifier

```python
RandomForestClassifier(
    n_estimators = 100,   # 100 decision trees
    random_state = 42,    # Reproducibility
    n_jobs       = -1     # Use all CPU cores
)
```

### Why Random Forest?
- Handles non-linear decision boundaries
- Resistant to overfitting
- Provides feature importance scores
- Robust to outliers

---

## 📈 Expected Results

| Metric   | Score    |
|----------|----------|
| Accuracy | ~96–100% |
| Precision| ~97%     |
| Recall   | ~97%     |
| F1-Score | ~97%     |

---

## 🚀 Setup & Run

### ✅ Prerequisites
- Python 3.8 or above
- pip package manager

### 💻 VS Code Commands

```bash
# Step 1: Open the project folder in VS Code
code iris_flower_classification

# Step 2: Open integrated terminal
# Press: Ctrl + ` (backtick)

# Step 3: Create a virtual environment
python -m venv venv

# Step 4: Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Step 5: Install dependencies
pip install -r requirements.txt

# Step 6: Run the script
python iris_classification.py
```

### 📦 Manual Installation (without venv)

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python iris_classification.py
```

---

## 🐙 GitHub Commands

```bash
# Initialize local git repository
git init

# Add all files
git add .

# First commit
git commit -m "feat: add Iris Flower Classification - CodeAlpha Task 1"

# Add your remote GitHub repo
git remote add origin https://github.com/23WH1A0530/iris-flower-classification.git

# Push to GitHub
git push -u origin main

# --- For future updates ---

# Check status
git status

# Stage changes
git add .

# Commit with message
git commit -m "fix: update README and add confusion matrix plot"

# Push
git push
```

---

## 🗂️ Libraries Used

| Library        | Purpose                                   |
|----------------|-------------------------------------------|
| `numpy`        | Numerical operations on arrays            |
| `pandas`       | DataFrame manipulation and analysis       |
| `matplotlib`   | Base plotting library                     |
| `seaborn`      | Statistical visualizations (pair plot, heatmap) |
| `scikit-learn` | ML model, metrics, and dataset            |

---

## 💬 Interview Explanation

**Q: What is the Iris dataset?**
> The Iris dataset contains 150 flower samples from 3 species (Setosa, Versicolor, Virginica), each described by 4 measurements: sepal length, sepal width, petal length, and petal width. It's a standard benchmark dataset for classification.

**Q: Why did you use Random Forest?**
> Random Forest is an ensemble method that builds multiple decision trees and aggregates their votes. It handles non-linear data well, is resistant to overfitting, and naturally provides feature importances — making it ideal for this multi-class classification task.

**Q: What is a train/test split and why 80/20?**
> We split data so the model trains on 80% of samples and is evaluated on unseen 20% test data. The 80/20 ratio is a widely-used convention that balances having enough data to train on while keeping a meaningful test set for evaluation.

**Q: What does the confusion matrix tell you?**
> It shows how many predictions were correct vs. incorrect for each class. The diagonal contains correct predictions; off-diagonal cells reveal misclassifications between species.

**Q: What is the classification report?**
> It provides precision, recall, and F1-score per class. Precision measures how many predicted positives are actually positive; recall measures how many actual positives were caught; F1 is the harmonic mean of both.

**Q: What did EDA reveal?**
> The pair plot showed that Setosa is linearly separable from the other two species. Versicolor and Virginica overlap slightly. The heatmap revealed that petal length and petal width are highly correlated (≈0.96), meaning they carry similar information.

---

## 👤 Author

**CodeAlpha Intern**
Machine Learning Internship — Task 1
[GitHub](https://github.com/23WH1A0530) • [LinkedIn](https://www.linkedin.com/in/lakshmi-kanta-sree-potula-178987297/)

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).# CodeAlpha_Iris_Flower_Classification
