# Gaming Addiction Classifier
# Goal: predict whether a gamer is addicted (addiction_binary = 1) or not (0).

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer

# 1. Load Data

df = pd.read_csv("data/gaming_addiction.csv", keep_default_na=False, na_values="")
print(f"rows, columns: {df.shape}")
print(f"")
print(df.head())
print(df['subscription_status'])
print("Addicted:", f"{df['addiction_binary'].mean():.1%}")


# 2. Choose features (X) and target (y)
drop_cols = ["user_id", "addiction_score", "addiction_severity",
             "behavioral_cluster", "burnout_probability", "mental_health_risk_score"]

target = 'addiction_binary'
X = df.drop(columns=drop_cols + [target])
y = df[target]


# 3. Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# 4. Prepare the columns
numeric_cols = X.select_dtypes(include=np.number).columns
text_cols = X.select_dtypes(exclude=np.number).columns

prepare = ColumnTransformer([
    ("numbers", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_cols),
    ("text", OneHotEncoder(handle_unknown="ignore"), text_cols),
], verbose_feature_names_out=False)


# 5. Compare models using cross-validation
models = {
    "logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "gradient boosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
    "SVM": SVC(random_state=42, probability=True),
    "XGBoost": XGBClassifier(random_state=42, n_estimators=100),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
select_by = "f1"

results = {}
for name, model in models.items():
    pipeline = make_pipeline(prepare, model)
    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=metrics)
    results[name] = {m: scores[f"test_{m}"].mean() for m in metrics}

results = pd.DataFrame(results).T.sort_values(select_by, ascending=False)
print("\nCross-validated scores:")
print(results.round(3).to_string())

best_name = results.index[0]
print(f"\nBest model (by {select_by}):", best_name)


# 6. Test the best model on the unseen test set
best = make_pipeline(prepare, models[best_name])
best.fit(X_train, y_train)

print("\nTest set results:")
print(classification_report(y_test, best.predict(X_test),
                            target_names=["Not addicted", "Addicted"]))
print("ROC-AUC:",round(roc_auc_score(y_test, best.predict_proba(X_test)[:,1]),3))

ConfusionMatrixDisplay.from_estimator(best, X_test, y_test,
                                      display_labels=["Not addicted", "Addicted"],cmap="Blues")
plt.title(f"Confusion matrix: {best_name}")
plt.savefig("confusion_matrix.png", dpi=200)
plt.close()
plt.show()


# 7. Which features matter most?
classifier = best[-1]
feature_names = best[0].get_feature_names_out()

if hasattr(classifier, "coef_"):  # Logistic Regression
    importance = abs(classifier.coef_[0])
elif hasattr(classifier, "feature_importances_"):  # tree models
    importance = classifier.feature_importances_
else:
    importance = None

if importance is not None:
    top = pd.Series(importance, index=feature_names).sort_values().tail(10)
    top.plot(kind="barh", title=f"Top 10 features: {best_name}")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=200)
    plt.close()
    print("\nTop 10 features:")
    print(top.sort_values(ascending=False).round(3).to_string())


# 8. Save the model and make a prediction
best.fit(X, y)  # final model learns from all the data
joblib.dump(best, "gaming_addiction_classifier.pkl")

player = X_test.iloc[[0]]  # one player (one row of data)
probability = best.predict_proba(player)[0, 1]
print(f"\nChance this player is addicted: {probability:.1%}")