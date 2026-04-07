import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from backend_automl import train_automl

# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def clf_df():
    X, y = make_classification(
        n_samples=300, n_features=10, n_informative=6,
        n_classes=2, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    df["target"] = y
    return df

@pytest.fixture
def imbalanced_clf_df():
    X, y = make_classification(
        n_samples=300, n_features=8, weights=[0.9, 0.1],
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["target"] = y
    return df

@pytest.fixture
def reg_df():
    X, y = make_regression(
        n_samples=300, n_features=8, noise=10, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["target"] = y
    return df

@pytest.fixture
def mixed_df():
    """Numeric + categorical columns"""
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(20, 60, 200).astype(float),
        "salary": np.random.normal(50000, 10000, 200),
        "city": np.random.choice(["Delhi", "Mumbai", "Pune"], 200),
        "grade": np.random.choice(["A", "B", "C", "D", "E", "F"], 200),
        "target": np.random.randint(0, 2, 200),
    })
    return df

@pytest.fixture
def missing_df():
    np.random.seed(0)
    df = pd.DataFrame(np.random.randn(200, 5), columns=[f"f{i}" for i in range(5)])
    df.iloc[::10, 0] = np.nan   # 10% missing in f0
    df.iloc[::5, 2]  = np.nan   # 20% missing in f2
    df["target"] = (df["f1"] > 0).astype(int)
    return df

# ── Core Return Shape ─────────────────────────────────────

def test_returns_three_values(clf_df):
    result = train_automl(clf_df, "target", task="classification")
    assert len(result) == 3, "Expected (model, score, model_name)"

def test_model_name_is_string(clf_df):
    _, _, name = train_automl(clf_df, "target", task="classification")
    assert isinstance(name, str) and len(name) > 0

# ── Classification Metrics ────────────────────────────────

def test_clf_accuracy_threshold(clf_df):
    model, score, _ = train_automl(clf_df, "target", task="classification")
    X = clf_df.drop(columns=["target"])
    y = clf_df["target"]
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    assert acc >= 0.75, f"Accuracy too low: {acc:.3f}"

def test_clf_f1_weighted(clf_df):
    model, _, _ = train_automl(clf_df, "target", task="classification")
    X = clf_df.drop(columns=["target"])
    y = clf_df["target"]
    preds = model.predict(X)
    f1 = f1_score(y, preds, average="weighted")
    assert f1 >= 0.70, f"F1 too low: {f1:.3f}"

def test_clf_roc_auc(clf_df):
    model, _, _ = train_automl(clf_df, "target", task="classification")
    X = clf_df.drop(columns=["target"])
    y = clf_df["target"]
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
    else:
        # fallback to decision function for SVC
        scores = model.decision_function(X)
        auc = roc_auc_score(y, scores)
    
    assert auc >= 0.80, f"AUC too low: {auc:.3f}"

def test_imbalanced_f1_not_trivial(imbalanced_clf_df):
    """Catches models that predict majority class always"""
    model, _, _ = train_automl(imbalanced_clf_df, "target", task="classification")
    X = imbalanced_clf_df.drop(columns=["target"])
    y = imbalanced_clf_df["target"]
    preds = model.predict(X)
    f1 = f1_score(y, preds, average="macro")
    assert f1 >= 0.50, f"Model collapsed to majority class, macro F1: {f1:.3f}"

# ── Regression Metrics ────────────────────────────────────

def test_reg_r2_threshold(reg_df):
    model, score, _ = train_automl(reg_df, "target", task="regression")
    X = reg_df.drop(columns=["target"])
    y = reg_df["target"]
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    assert r2 >= 0.70, f"R² too low: {r2:.3f}"

def test_reg_rmse_reasonable(reg_df):
    model, _, _ = train_automl(reg_df, "target", task="regression")
    X = reg_df.drop(columns=["target"])
    y = reg_df["target"]
    preds = model.predict(X)
    rmse = mean_squared_error(y, preds) ** 0.5
    baseline_rmse = y.std()  # predicting mean baseline
    assert rmse < baseline_rmse, f"Model worse than mean baseline. RMSE: {rmse:.2f}, Baseline: {baseline_rmse:.2f}"

def test_reg_mae(reg_df):
    model, _, _ = train_automl(reg_df, "target", task="regression")
    X = reg_df.drop(columns=["target"])
    y = reg_df["target"]
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    assert mae < y.std(), f"MAE too high: {mae:.2f}"

# ── Robustness ────────────────────────────────────────────

def test_handles_missing_values(missing_df):
    model, score, _ = train_automl(missing_df, "target", task="classification")
    assert score > 0.0, "Pipeline failed on missing data"

def test_handles_mixed_dtypes(mixed_df):
    model, score, _ = train_automl(mixed_df, "target", task="classification")
    assert score > 0.0, "Pipeline failed on mixed dtypes"

def test_cv_score_matches_refit(clf_df):
    """CV score should be <= train score (no data leakage smoke test)"""
    model, cv_score, _ = train_automl(clf_df, "target", task="classification")
    X = clf_df.drop(columns=["target"])
    y = clf_df["target"]
    train_acc = accuracy_score(y, model.predict(X))
    assert train_acc >= cv_score - 0.05, "CV score suspiciously higher than train score"

# ── Serialization ─────────────────────────────────────────

def test_model_serializable(clf_df, tmp_path):
    import joblib
    model, _, _ = train_automl(clf_df, "target", task="classification")
    path = tmp_path / "model.pkl"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    X = clf_df.drop(columns=["target"])
    assert list(model.predict(X))