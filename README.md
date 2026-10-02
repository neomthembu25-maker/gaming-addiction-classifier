# 🎮 Gaming Addiction Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A machine learning project that predicts whether a gamer is at risk of addiction from their gaming habits, psychological traits and lifestyle. Five models were compared with cross-validation. The best, **Logistic Regression**, reached **94% accuracy** and **0.83 F1** in cross-validation, and caught **7 of 8** addicted players on the unseen test set.

---

## 📌 Overview

Excessive gaming is linked to poorer mental health, falling academic or work performance, and social isolation. This project asks: **can behavioural and wellbeing indicators flag addiction risk early?**

**Data:** public Kaggle dataset ([add link]) with 250 players, of whom 16.8% are labelled addicted.
**Target:** `addiction_binary` (1 = addicted, 0 = not addicted).
**Features:** 42 columns covering:

| Group | Examples |
|-------|----------|
| Demographics | age, gender, country, occupation, income level |
| Gaming behaviour | daily playtime, weekly sessions, late-night hours, platform, genre |
| Spending | monthly spend, in-game purchases, loot box openings |
| Psychological | stress, loneliness, self-control, impulsiveness, dopamine dependency |
| Lifestyle and performance | sleep, exercise, caffeine, social time, GPA / performance, absenteeism |

---

## ⚙️ Method

1. **Remove leakage.** `addiction_binary` is simply `addiction_score > 50`, so the score and the columns derived from it (`addiction_severity`, `behavioral_cluster`, `burnout_probability`, `mental_health_risk_score`) are dropped, along with `user_id`.
2. **Load carefully.** The value `"None"` in `subscription_status` is a real category, so it is kept instead of being read as missing.
3. **Split.** 80% train (200 players) and 20% test (50 players), stratified so both parts keep the same share of addicted players.
4. **Prepare inside a pipeline.** Numeric columns: median imputation, then standardisation. Text columns: one-hot encoding. Everything is fitted on training data only, so nothing leaks from the test set.
5. **Compare models with 5-fold stratified cross-validation** on the training set: Logistic Regression, Random Forest, Gradient Boosting, SVM and XGBoost. Models are ranked by F1, because the classes are imbalanced and accuracy alone would be misleading.
6. **Test once.** The best model is evaluated a single time on the held-out test set, then refitted on all the data and saved.

---

## 🏆 Results

### Cross-validation (training set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| **Logistic Regression** | **0.945** | 0.848 | **0.829** | **0.832** | **0.976** |
| XGBoost | 0.925 | 0.876 | 0.681 | 0.735 | 0.955 |
| Gradient Boosting | 0.895 | 0.719 | 0.648 | 0.674 | 0.944 |
| SVM | 0.910 | **0.920** | 0.533 | 0.641 | 0.975 |
| Random Forest | 0.885 | 0.693 | 0.381 | 0.488 | 0.951 |

Logistic Regression leads on every metric except precision, so the choice of model does not depend on which metric is used. The simplest model beat the more complex ones on this small dataset.

### Test set (50 players, 8 addicted)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Not addicted | 0.98 | 0.95 | 0.96 |
| Addicted | 0.78 | 0.88 | 0.82 |

Accuracy **94%** · ROC-AUC **0.988**. The model caught 7 of the 8 addicted players, missed 1, and raised 2 false alarms.

![Confusion matrix](https://github.com/neomthembu25-maker/gaming-addiction-classifier/blob/main/Results/confusion_matrix.png)

---

## 🔍 What drives the predictions

The strongest predictors are the largest standardised coefficients in the Logistic Regression model:

| Rank | Feature | Coefficient size |
|------|---------|------------------|
| 1 | Self-control score | 1.60 |
| 2 | Late-night gaming hours | 1.53 |
| 3 | Impulsiveness score | 1.16 |
| 4 | Dopamine dependency index | 0.92 |
| 5 | Daily playtime hours | 0.85 |
| 6 | Sleep hours | 0.77 |

Psychological traits (self-control, impulsiveness, dopamine dependency) and sleep-disrupting late-night play matter at least as much as raw hours played.

![Feature importance](feature_importance.png)

---

## 🚀 Getting Started

```bash
git clone https://github.com/neomthembu25-maker/[repo-name].git
cd [repo-name]
pip install -r requirements.txt
```

Put the Kaggle CSV at `data/gaming_addiction.csv`, then:

```bash
python model.py      # trains, evaluates, saves the model and charts
python predict.py    # example predictions for one player and for a whole file
streamlit run app.py # interactive risk estimator in your browser
```

### Predicting for a new player

```python
import joblib, pandas as pd

model = joblib.load("gaming_addiction_classifier.pkl")
features = list(model.feature_names_in_)          # the 42 columns the model needs

df = pd.read_csv("data/gaming_addiction.csv", keep_default_na=False, na_values="")
player = df[features].iloc[[0]].copy()            # start from a real row
player["daily_playtime_hours"] = 9.0              # change what you want

print(model.predict_proba(player)[0, 1])          # probability of addiction
```

---

## 📁 Project Structure

```
├── data/gaming_addiction.csv
├── model.py        # training, comparison, evaluation, saving
├── predict.py      # single and batch predictions
├── app.py          # Streamlit app
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

- **Small dataset.** With 250 players and only 8 addicted in the test set, one different prediction moves recall by about 12 points. The cross-validated scores are the more reliable result.
- **No hyperparameter tuning**, and class imbalance (17% addicted) is not yet handled with class weights.
- **Importance is not causation.** The model uses these features to predict, which does not mean they cause addiction.
- **Possible remaining leakage.** Columns such as `churn_probability` and `productivity_drop_percent` may be consequences of addiction rather than early warning signs.
- **Not a diagnostic tool.** This is an educational project trained on a public dataset, not a clinical screening instrument.

---

## 🔮 Future Work

- Tune hyperparameters and try class weights
- Re-run without the borderline features to check for leakage
- Add SHAP values to explain individual predictions
- Validate on a second dataset

---

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · matplotlib · joblib · Streamlit

---

## 👤 Author

**Neo Mthembu**: BCom Statistics student

[GitHub](https://github.com/neomthembu25-maker) · [Portfolio](https://datascienceportfol.io/neomthembu25) · [LinkedIn](https://linkedin.com/in/neo-mthembu-902b4621a)
