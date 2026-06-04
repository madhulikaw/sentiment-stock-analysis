# 📈 GME Meme Stock Prediction via Alternative Reddit Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Data Source](https://img.shields.io/badge/Data-Reddit%20%26%20Yahoo%20Finance-red.svg)]()

A production-grade quantitative data science pipeline designed to extract, clean, and model alternative social media metrics alongside historical equity data. This project evaluates whether retail investor sentiment and crowd attention volume from subreddits like r/wallstreetbets serve as directional leading market indicators for highly volatile "meme stocks" like GameStop (`GME`).

---

## 🎯 Core Project Performance Summary

Unlike traditional machine learning problems where 90%+ scores are common, financial markets are heavily bound to a random walk. This framework enforces strict real-world constraints (zero look-ahead bias and a chronological split) to extract true predictive alpha.

| Phase Milestone | Target/Baseline Metric | Optimized Model Performance | Key Strategic Takeaway |
| :--- | :---: | :---: | :--- |
| **Directional Accuracy** | 50.00% (Coin-Flip Baseline) | **66.67%** (Random Forest) | Successfully anticipated 4 out of 6 unseen market days. |
| **Feature Driving Order** | N/A | **Yesterday_Posts** (>25%) | Retail crowd attention volume out-predicted explicit sentiment polarity. |
| **Model Persistence** | Dynamic Execution | **Static Production Artifact** | Model weights successfully preserved into reusable `.joblib` format. |

---

## 📂 Repository Structural Layout

```text
sentiment-stock-analysis/
│
├── data/
│   └── gme_merged_analysis.csv       # Unified dataset combining Reddit text data and stock market metrics
│
├── models/
│   └── gme_random_forest_model.joblib # Trained and regularized serialized model binary
│
├── notebooks/
│   ├── 07_predictive_modeling.ipynb  # Phase 7: Regularized Machine Learning Core
│   └── 08_model_evaluation.ipynb     # Phase 8: Confusion Matrix, Importances, and Exports
│
├── GME_Model_Evaluation_Report.pdf   # Auto-generated programmatic Executive Report
└── README.md                         # Project Documentation Landing Page