# Iris-flower-dataset
# Iris Flower Classification

# Iris Flower Dataset Analysis

This project demonstrates an analysis of the Iris Flower dataset using a Jupyter Notebook. The analysis includes exploratory data visualization, feature correlation, and insights into the dataset.

## Dataset
The dataset contains measurements for 150 Iris flowers, covering the following species:
- Setosa
- Versicolor
- Virginica

### Features
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

## Contents
- **iris-flower-dataset.ipynb**: The Jupyter Notebook with the analysis.
- **README.md**: This file, explaining the project structure and how to run it.

## Installation

### Requirements
- Python 3.x
- Jupyter Notebook
- Libraries: pandas, matplotlib, seaborn, numpy

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Iris-Flower-Analysis.git
   cd Iris-Flower-Analysis


## Files
- `IRIS.csv`: Dataset file
- `decision_tree_model.pkl`: Saved model file
- `submission.csv`: Submission results

## Key Metrics
- Decision Tree Accuracy: **95.55%**

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python train_model.py

---

### **9. Experiment with Hyperparameter Tuning**
Improve your model's performance using `GridSearchCV`.

#### Code:
```python
from sklearn.model_selection import GridSearchCV

# Define hyperparameters
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Update the model with the best parameters
best_model = grid_search.best_estimator_

