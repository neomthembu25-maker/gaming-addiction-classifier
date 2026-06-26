"""
GAMING ADDICTION CLASSIFIER - Simplified Version
Binary Classification to Predict Gaming Addiction Risk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

sns.set_style("white")

# ============================================================
# 1. LOAD & EXPLORE DATA
# ============================================================
print("=" * 60)
print("GAMING ADDICTION CLASSIFIER")
print("=" * 60)

df = pd.read_csv(r'C:\Users\36321389\Downloads\gaming_addiction.csv')
print(f"\nShape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

print("\nTarget Distribution:")
print(df['addiction_binary'].value_counts())
print(f"Addicted: {df['addiction_binary'].mean() * 100:.1f}%")

# ============================================================
# 2. PREPROCESS DATA
# ============================================================
print("\n" + "=" * 60)
print("PREPROCESSING")
print("=" * 60)

df_clean = df.copy()

# Drop columns with >30% missing
missing_threshold = 0.3
missing_cols = df_clean.columns[df_clean.isnull().mean() > missing_threshold]
df_clean = df_clean.drop(columns=missing_cols)
print(f"Dropped: {missing_cols.tolist()}")

# Impute missing values
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
categorical_cols = df_clean.select_dtypes(include=['object']).columns

for col in numeric_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

for col in categorical_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# Drop leakage columns
leakage_cols = ['addiction_score', 'burnout_probability',
                'mental_health_risk_score', 'behavioral_cluster', 'addiction_severity']
df_clean = df_clean.drop(columns=[c for c in leakage_cols if c in df_clean.columns])

# Define X and y
target = 'addiction_binary'
X = df_clean.drop(columns=[target])
y = df_clean[target]

# ✅ REMOVE user_id from features (FIX)
if 'user_id' in X.columns:
    X = X.drop(columns=['user_id'])
    print("✅ Dropped 'user_id' from features")

# Identify feature types
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric: {len(numeric_features)}, Categorical: {len(categorical_features)}")

# Create preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================
# 3. TRAIN MODELS
# ============================================================
print("\n" + "=" * 60)
print("TRAINING MODELS")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining: {name}")

    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pipeline': pipeline
    }

    print(f"  Accuracy: {results[name]['accuracy']:.4f}")
    print(f"  F1-Score: {results[name]['f1']:.4f}")
    print(f"  ROC-AUC:  {results[name]['roc_auc']:.4f}")

# ============================================================
# 4. COMPARE MODELS
# ============================================================
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame({
    name: {
        'Accuracy': results[name]['accuracy'],
        'Precision': results[name]['precision'],
        'Recall': results[name]['recall'],
        'F1-Score': results[name]['f1'],
        'ROC-AUC': results[name]['roc_auc']
    }
    for name in results.keys()
}).T

print(comparison_df.round(4))

# Best model
best_name = comparison_df['F1-Score'].idxmax()
best_pipeline = results[best_name]['pipeline']
print(f"\n🏆 Best Model: {best_name}")
print(f"   F1-Score: {comparison_df.loc[best_name, 'F1-Score']:.4f}")
print(f"   ROC-AUC:  {comparison_df.loc[best_name, 'ROC-AUC']:.4f}")

# ============================================================
# 5. VISUALIZE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("VISUALIZATIONS")
print("=" * 60)

# ROC Curves
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (name, result) in enumerate(results.items()):
    if i < 6:
        y_proba = result['pipeline'].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[i].plot(fpr, tpr, label=f'{name} (AUC={result["roc_auc"]:.3f})')
        axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(name)
        axes[i].legend()
        axes[i].grid(True)

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300)
plt.show()

# Bar Chart
fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
comparison_df[metrics].plot(kind='bar', ax=ax)
ax.set_title('Model Performance Comparison', fontsize=14)
ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.legend(loc='lower right')
ax.set_xticklabels(comparison_df.index, rotation=45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()

# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

if hasattr(best_pipeline.named_steps['classifier'], 'feature_importances_'):
    # Get feature names
    feature_names = numeric_features.copy()
    cat_encoder = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    if hasattr(cat_encoder, 'get_feature_names_out'):
        feature_names.extend(cat_encoder.get_feature_names_out(categorical_features))

    importances = best_pipeline.named_steps['classifier'].feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Features:")
    print(importance_df.head(10).to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    top = importance_df.head(15)
    ax.barh(top['feature'], top['importance'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top 15 Features - {best_name}')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.show()

# ============================================================
# 7. FINAL EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

y_pred = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Addicted', 'Addicted']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Addicted', 'Addicted'],
            yticklabels=['Not Addicted', 'Addicted'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix - {best_name}')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# ============================================================
# 8. SAVE MODEL
# ============================================================
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

joblib.dump(best_pipeline, 'gaming_addiction_classifier.pkl')
print("✅ Model saved as 'gaming_addiction_classifier.pkl'")
print("   model = joblib.load('gaming_addiction_classifier.pkl')")

# ============================================================
# 9. MAKE PREDICTION (NO user_id needed!)
# ============================================================
print("\n" + "=" * 60)
print("MAKE PREDICTION")
print("=" * 60)


def predict_player(model, features_df):
    pred = model.predict(features_df)[0]
    prob = model.predict_proba(features_df)[0][1]
    return pred, prob


# Sample player (NO user_id needed - it was dropped)
sample = pd.DataFrame({
    'age': [25],
    'gender': ['Male'],
    'country': ['USA'],
    'occupation': ['Employed'],
    'income_level': ['Middle'],
    'years_gaming': [13],
    'preferred_genre': ['RPG'],
    'platform': ['PC'],
    'device_type': ['Laptop'],
    'rank_tier': ['Gold'],
    'daily_playtime_hours': [8.5],
    'weekly_play_sessions': [10],
    'late_night_sessions_hours': [3.0],
    'weekend_playtime_hours': [12.0],
    'consecutive_hours_max': [15.0],
    'multiplayer_ratio': [0.5],
    'toxic_chat_reports': [0],
    'rage_quit_frequency': [0],
    'in_game_purchases': [0],
    'monthly_spending_usd': [30.0],
    'lootbox_openings': [5],
    'subscription_status': ['None'],
    'stress_score': [8.0],
    'loneliness_score': [7.0],
    'dopamine_dependency_index': [6.0],
    'self_control_score': [4.0],
    'impulsiveness_score': [5.0],
    'anxiety_level': [7.0],
    'depression_indicator': [1],
    'emotional_stability': [4.0],
    'sleep_hours': [5.0],
    'exercise_frequency_per_week': [1],
    'caffeine_intake_cups_day': [3],
    'social_interaction_hours': [2.0],
    'relationship_status': ['Single'],
    'gpa_or_performance_score': [3.0],
    'missed_deadlines': [3],
    'productivity_drop_percent': [15.0],
    'absenteeism_days': [5],
    'internet_speed_mbps': [80.0],
    'screen_time_total_hours': [10.0],
    'churn_probability': [0.5]
})

pred, prob = predict_player(best_pipeline, sample)

print("\nPREDICTION RESULT")
print("-" * 40)
print(f"Prediction: {'ADDICTED' if pred == 1 else 'NOT ADDICTED'}")
print(f"Probability: {prob:.2%}")
risk = 'HIGH' if prob > 0.7 else 'MODERATE' if prob > 0.5 else 'LOW'
print(f"Risk Level: {risk}")

# ============================================================
# 10. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
✅ Best Model: {best_name}
✅ Accuracy:  {comparison_df.loc[best_name, 'Accuracy']:.2%}
✅ F1-Score:  {comparison_df.loc[best_name, 'F1-Score']:.2%}
✅ ROC-AUC:   {comparison_df.loc[best_name, 'ROC-AUC']:.2%}

📁 Files Saved:
   • gaming_addiction_classifier.pkl
   • roc_curves.png
   • model_comparison.png
   • feature_importance.png
   • confusion_matrix.png
""")
print("=" * 60)