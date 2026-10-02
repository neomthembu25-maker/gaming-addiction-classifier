# Make predictions with the saved model.
# Run model.py first so gaming_addiction_classifier.pkl exists.

import joblib
import pandas as pd

model = joblib.load("gaming_addiction_classifier.pkl")

# The model needs one value for each of these columns (same names as training).
features = list(model.feature_names_in_)
print(f"The model expects {len(features)} columns")

# ------------------------------------------------------------------
# A. Predict for ONE player
# ------------------------------------------------------------------
# Typing all the columns by hand is tedious, so start from a real row
# and change the values you care about.
df = pd.read_csv("Downloads/gaming_addiction.csv", keep_default_na=False, na_values="")
player = df[features].iloc[[0]].copy()          # [[0]] keeps it as a one-row table

player["daily_playtime_hours"] = 9.0
player["self_control_score"] = 3.0
player["late_night_sessions_hours"] = 4.0

probability = model.predict_proba(player)[0, 1]  # chance of being addicted (0 to 1)
prediction = model.predict(player)[0]            # 1 = addicted, 0 = not addicted

print(f"\nPrediction:  {'ADDICTED' if prediction == 1 else 'NOT ADDICTED'}")
print(f"Probability: {probability:.1%}")

# ------------------------------------------------------------------
# B. Predict for MANY players from a CSV file
# ------------------------------------------------------------------
# The file needs the same columns as the training data (extra columns are fine).
new_players = pd.read_csv("Downloads/gaming_addiction.csv", keep_default_na=False, na_values="")

new_players["addiction_probability"] = model.predict_proba(new_players[features])[:, 1]
new_players["predicted_addicted"] = model.predict(new_players[features])

print(new_players[["user_id", "addiction_probability", "predicted_addicted"]].head())
new_players.to_csv("predictions.csv", index=False)
print("\nSaved predictions.csv")
