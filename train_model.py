import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

DATA_FILE  = 'hand_data.csv'
MODEL_FILE = 'hand_model.pkl'

df = pd.read_csv(DATA_FILE, header=None)
print(f"Loaded {len(df)} samples")
print("Counts per label:")
print(df[0].value_counts().sort_index())

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
#RandomForrestClassifier (Enhanced Decision tree)
clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
#n_estimators - the number trees in the forest
#max_depth - the maximum depth of the tree( how many grandchildren should it stop at)
#random_state - controls the randomness

scores = cross_val_score(clf, X, y, cv=5)
print(f"Cross-validation accuracy: {scores.mean():.1%} (+/- {scores.std():.1%})")

clf.fit(X, y)
joblib.dump(clf, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")