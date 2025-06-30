# 📦 End-to-End Spam Detection Pipeline using DVC & Scikit-learn

This project is an end-to-end machine learning pipeline for spam detection using the `spam.csv` dataset. The entire workflow is modular, version-controlled with DVC, and powered by various classical machine learning models using `scikit-learn`.

---

## 🚀 Features

- ✅ Modular code structure (data loading, preprocessing, training, evaluation)
- ✅ Experiment tracking with [DVC](https://dvc.org/)
- ✅ Support for multiple classifiers:
  - Random Forest
  - K-Nearest Neighbors
  - Support Vector Classifier
  - Gradient Boosting
  - Logistic Regression
  - And more!
- ✅ Cross-validation & metrics tracking
- ✅ Configurable pipeline stages

---

## 🗂 Project Structure

```bash
.
├── data/                  # Raw and processed data (tracked by DVC)
│   └── spam.csv           # Dataset file
├── src/                   # Modular Python code
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── models/                # Saved model files (optional, tracked by DVC)
├── metrics/               # Evaluation metrics (e.g., accuracy, f1-score)
├── dvc.yaml               # DVC pipeline definition
├── params.yaml            # Model parameters and config
├── requirements.txt       # Python dependencies
└── README.md              # Project overview
