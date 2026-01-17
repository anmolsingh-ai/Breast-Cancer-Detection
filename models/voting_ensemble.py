import joblib
from pathlib import Path

import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def load_data():
    repo_root = Path(__file__).resolve().parent.parent
    raw_path = repo_root / 'data' / 'raw' / 'Breast_Cancer.csv'
    processed_path = repo_root / 'data' / 'processed' / 'processed_breast_cancer.csv'

    df_raw = pd.read_csv(raw_path)
    df_raw['Status'] = df_raw['Status'].map({'Alive': 1, 'Dead': 0})
    y = df_raw['Status']

    X = pd.read_csv(processed_path)

    # Ensure indices align
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y


def build_ensemble():
    estimators = [
        ('lr', LogisticRegression(solver='liblinear', random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ]

    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    return ensemble


def train_and_evaluate(save_model=True):
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_ensemble()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {acc:.4f}')
    print('Classification report:')
    print(report)

    if save_model:
        out_dir = Path(__file__).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / 'voting_model.joblib'
        joblib.dump(model, model_path)
        print(f'Saved model to: {model_path}')

    return model


if __name__ == '__main__':
    train_and_evaluate()
