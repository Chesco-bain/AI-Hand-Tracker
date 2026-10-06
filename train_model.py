import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

DATA_FILE  = 'hand_data.csv'
MODEL_FILE = 'hand_model.pkl'

df = pd.read_csv(DATA_FILE, header=None)
print(f"Loaded {len(df)} samples")
print("Counts per number:")
print(df[0].value_counts().sort_index())

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)

scores = cross_val_score(clf, X, y, cv=5)
print(f"Cross-validation accuracy: {scores.mean():.1%} (+/- {scores.std():.1%})")

clf.fit(X, y)
joblib.dump(clf, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")