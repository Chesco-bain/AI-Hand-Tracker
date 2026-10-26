import pandas as pd
import joblib
import numpy as np
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

def augment_data(X, y):
    print("Augmenting data to handle lighting/jitter...")
    X_aug = [X]
    y_aug = [y]
    for _ in range(4):
        X_noisy = X + np.random.normal(0, 0.02, X.shape)
        scale   = np.random.uniform(0.95, 1.05, (X.shape[0], 1))
        X_scaled = X * scale
        X_aug.extend([X_noisy, X_scaled])
        y_aug.extend([y, y])
    return np.vstack(X_aug), np.concatenate(y_aug)

X_train, y_train = augment_data(X, y)
print(f"Total samples after augmentation: {len(X_train)}")

clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)

# Guard cv against tiny classes
min_class = pd.Series(y_train).value_counts().min()
n_folds = max(2, min(5, min_class))
print(f"Running {n_folds}-fold cross-validation...")
scores = cross_val_score(clf, X_train, y_train, cv=n_folds)
print(f"Cross-validation accuracy: {scores.mean():.1%} (+/- {scores.std():.1%})")

clf.fit(X_train, y_train)
joblib.dump(clf, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")