# 🎮 Gaming Addiction Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red.svg)](https://github.com/yourusername/gaming-addiction-classifier)

> Machine Learning model to predict gaming addiction risk using behavioral, psychological, and lifestyle features

---

## 📊 Overview

This project builds a **binary classification model** that predicts whether a gamer is at risk of addiction based on:

- **Demographics** (age, gender, country, occupation, income)
- **Gaming Behavior** (playtime, sessions, late-night gaming, platform)
- **Psychological Factors** (stress, loneliness, self-control, impulsiveness)
- **Lifestyle Indicators** (sleep, exercise, social interaction, caffeine intake)
- **Academic/Work Performance** (GPA, productivity, missed deadlines)

### 🎯 Problem Statement

Gaming addiction is a growing concern worldwide. Early detection can help prevent serious mental health issues, academic decline, and social isolation. This model provides a data-driven approach to identify at-risk individuals early.

---

## 🚀 Key Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | **94.0%** | **77.8%** | **87.5%** | **82.4%** | **97.0%** |
| Logistic Regression | 92.0% | 75.0% | 75.0% | 75.0% | 98.8% |
| Gradient Boosting | 92.0% | 75.0% | 75.0% | 75.0% | 95.5% |
| SVM | 90.0% | 80.0% | 50.0% | 61.5% | 96.1% |
| Random Forest | 88.0% | 66.7% | 50.0% | 57.1% | 91.7% |

**🏆 Best Model:** XGBoost

---

## 🔑 Key Findings

### Top 10 Predictors of Gaming Addiction

| Rank | Feature | Importance | Insight |
|------|---------|------------|---------|
| 1 | **Daily Playtime Hours** | 17.7% | Most important predictor |
| 2 | **GPA/Performance Score** | 15.2% | Academic decline is a red flag |
| 3 | **Low Income** | 6.1% | Financial stress increases risk |
| 4 | **Self-Control Score** | 5.6% | Lower self-control = higher risk |
| 5 | **Dopamine Dependency Index** | 5.6% | Reward-seeking behavior |
| 6 | **Late Night Sessions** | 4.6% | Disrupted sleep patterns |
| 7 | **Impulsiveness Score** | 4.5% | Impulsive behavior increases risk |
| 8 | **Monthly Spending** | 3.6% | Financial investment in games |
| 9 | **Loneliness Score** | 3.5% | Social isolation is a risk factor |
| 10 | **Screen Time Total** | 3.2% | Overall digital consumption |

### 💡 Actionable Insights

Based on these findings, here are practical interventions:

Clustering Dataset Suitability
Here's a complete, professional README.md for your Gaming Addiction Classifier project:

📄 README.md
markdown
# 🎮 Gaming Addiction Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red.svg)](https://github.com/yourusername/gaming-addiction-classifier)

> Machine Learning model to predict gaming addiction risk using behavioral, psychological, and lifestyle features

---

## 📊 Overview

This project builds a **binary classification model** that predicts whether a gamer is at risk of addiction based on:

- **Demographics** (age, gender, country, occupation, income)
- **Gaming Behavior** (playtime, sessions, late-night gaming, platform)
- **Psychological Factors** (stress, loneliness, self-control, impulsiveness)
- **Lifestyle Indicators** (sleep, exercise, social interaction, caffeine intake)
- **Academic/Work Performance** (GPA, productivity, missed deadlines)

### 🎯 Problem Statement

Gaming addiction is a growing concern worldwide. Early detection can help prevent serious mental health issues, academic decline, and social isolation. This model provides a data-driven approach to identify at-risk individuals early.

---

## 🚀 Key Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | **94.0%** | **77.8%** | **87.5%** | **82.4%** | **97.0%** |
| Logistic Regression | 92.0% | 75.0% | 75.0% | 75.0% | 98.8% |
| Gradient Boosting | 92.0% | 75.0% | 75.0% | 75.0% | 95.5% |
| SVM | 90.0% | 80.0% | 50.0% | 61.5% | 96.1% |
| Random Forest | 88.0% | 66.7% | 50.0% | 57.1% | 91.7% |

**🏆 Best Model:** XGBoost

---

## 🔑 Key Findings

### Top 10 Predictors of Gaming Addiction

| Rank | Feature | Importance | Insight |
|------|---------|------------|---------|
| 1 | **Daily Playtime Hours** | 17.7% | Most important predictor |
| 2 | **GPA/Performance Score** | 15.2% | Academic decline is a red flag |
| 3 | **Low Income** | 6.1% | Financial stress increases risk |
| 4 | **Self-Control Score** | 5.6% | Lower self-control = higher risk |
| 5 | **Dopamine Dependency Index** | 5.6% | Reward-seeking behavior |
| 6 | **Late Night Sessions** | 4.6% | Disrupted sleep patterns |
| 7 | **Impulsiveness Score** | 4.5% | Impulsive behavior increases risk |
| 8 | **Monthly Spending** | 3.6% | Financial investment in games |
| 9 | **Loneliness Score** | 3.5% | Social isolation is a risk factor |
| 10 | **Screen Time Total** | 3.2% | Overall digital consumption |

### 💡 Actionable Insights

Based on these findings, here are practical interventions:
### ⏰ Daily Playtime Hours
→ Implement playtime limits or break reminders after 2 hours

### 📚 GPA/Performance Score
→ Monitor academic performance; partner with schools

### 🧘 Self-Control Score
→ Offer self-control training or mindfulness features in-game

### 🌙 Late Night Sessions
→ Add "bedtime mode" that discourages gaming after 11 PM

### 🎯 Dopamine Dependency
→ Diversify game rewards to reduce dependency on single mechanics

### 👥 Loneliness Score
→ Encourage social connections and positive in-game interactions 

---

## 📂 Project Structure
