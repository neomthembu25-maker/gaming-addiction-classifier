# 🎮 Gaming Addiction Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> A binary classification model that predicts gaming addiction risk from behavioural, psychological and lifestyle features. The best model, XGBoost, reaches **94% accuracy** and **87.5% recall** on the held-out test set.

---

## 📌 Overview

Gaming addiction is a growing concern, linked to poorer mental health, academic decline and social isolation. This project asks: **can a player's habits and wellbeing indicators flag addiction risk early?**

The model uses 40+ features across five groups:

| Group | Examples |
|-------|----------|
| **Demographics** | age, gender, country, occupation, income level |
| **Gaming behaviour** | daily playtime, weekly sessions, late-night hours, platform, genre |
| **Spending** | monthly spend, in-game purchases, loot box openings |
| **Psychological** | stress, loneliness, self-control, impulsiveness, anxiety |
| **Lifestyle and performance** | sleep, exercise, caffeine, social time, GPA/performance, missed deadlines |

**Dataset:** [add source / link] · **Rows:** [n] · **Train/test split:** [e.g. 80/20, n test = ?]

---

## 🏆 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| **XGBoost** | **94.0%** | 77.8% | **87.5%** | **82.4%** | 97.0% |
| Logistic Regression | 92.0% | 75.0% | 75.0% | 75.0% | **98.8%** |
| Gradient Boosting | 92.0% | 75.0% | 75.0% | 75.0% | 95.5% |
| SVM | 90.0% | 80.0% | 50.0% | 61.5% | 96.1% |
| Random Forest | 88.0% | 66.7% | 50.0% | 57.1% | 91.7% |

XGBoost was chosen as the best model because it has the highest accuracy, recall and F1. Recall matters most here, since missing an at-risk player is costlier than a false alarm. Logistic Regression has the highest ROC-AUC, so it remains a strong, interpretable baseline.

---

## 🔍 Key Findings

Top predictors from XGBoost feature importance:

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | Daily playtime hours | 17.7% |
| 2 | GPA / performance score | 15.2% |
| 3 | Low income | 6.1% |
| 4 | Self-control score | 5.6% |
| 5 | Dopamine dependency index | 5.6% |
| 6 | Late-night sessions | 4.6% |
| 7 | Impulsiveness score | 4.5% |
| 8 | Monthly spending | 3.6% |
| 9 | Loneliness score | 3.5% |
| 10 | Total screen time | 3.2% |

**Takeaways**
- Playtime and a drop in academic performance dominate the model's predictions.
- Self-control, impulsiveness and dopamine dependency show that psychological traits matter alongside raw hours played.
- Late-night play, spending and loneliness add smaller but consistent signal.

**Possible interventions** (suggested by the findings, not tested):
- Playtime limits or break reminders for heavy daily play
- A "bedtime mode" that discourages late-night sessions
- Self-control and mindfulness features, plus encouragement of social play
- Early check-ins with schools or employers when performance drops

---

## ⚠️ Limitations

- **Small test set.** Metrics come from a limited number of test samples, so a single misclassification shifts them noticeably. Cross-validation would give more reliable estimates.
- **Feature importance is not causation.** A high importance for income or GPA means the model uses them to predict, not that they cause addiction.
- **Possible leakage.** Some features (e.g. performance drop, churn probability) may be consequences of addiction rather than predictors of it. Review before any real-world use.
- **Not a diagnostic tool.** This is a learning project, not a clinical screening instrument.

---

## 🚀 Getting Started

```bash
git clone https://github.com/neomthembu25-maker/[repo-name].git
cd [repo-name]
pip install -r requirements.txt
python model.py          # trains the models and saves the best one
```

### Predict for a new player

```python
import joblib
import pandas as pd

model = joblib.load("gaming_addiction_classifier.pkl")

# One row containing every feature the model was trained on
sample = pd.DataFrame({
    "age": [25], "gender": ["Male"], "country": ["USA"],
    "occupation": ["Employed"], "income_level": ["Middle"],
    "daily_playtime_hours": [8.5], "late_night_sessions_hours": [3.0],
    "stress_score": [8.0], "loneliness_score": [7.0],
    "self_control_score": [4.0], "sleep_hours": [5.0],
    "gpa_or_performance_score": [3.0],
    # ... remaining features, see the full list in model.py
})

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print("Prediction:", "ADDICTED" if prediction == 1 else "NOT ADDICTED")
print(f"Risk score: {probability:.2%}")
```

---

## 🛠️ Tech Stack

Python · pandas · scikit-learn · XGBoost · matplotlib / seaborn · joblib

---

## 🔮 Future Work

- k-fold cross-validation and hyperparameter tuning
- SHAP values for more reliable, per-player explanations
- Check for leakage and test on a second dataset
- Simple web app (e.g. Streamlit) for live predictions

---

## 👤 Author

**Neo Mthembu**: BCom Statistics student

[GitHub](https://github.com/neomthembu25-maker) · [Portfolio](https://datascienceportfol.io/neomthembu25) · [LinkedIn](https://linkedin.com/in/neo-mthembu-902b4621a)
