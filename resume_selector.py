"""Resume selector v2: honest evaluation of a small text classifier.

Everything that learns from data (the vocabulary, the term weights) lives inside a scikit-learn Pipeline, so cross-validation
fits it on the training fold only. v1 built its vocabulary from all 125 resumes before splitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

MODELS = ("majority", "naive_bayes", "logreg", "naive_bayes_tuned", "logreg_tuned")
_TOKENS = r"(?u)\b[a-zA-Z][a-zA-Z+#.]{1,}\b"   # words of 2+ letters; keeps c++, c#, node.js style tokens


def load_resumes(path: str) -> pd.DataFrame:
    """The CSV is Latin-1 encoded. Label 1 = flagged, 0 = not flagged."""
    df = pd.read_csv(path, encoding="latin-1")
    df["label"] = (df["class"] == "flagged").astype(int)
    return df[["resume_id", "resume_text", "label"]]


def build_pipeline(name: str) -> Pipeline:
    if name == "majority":
        return Pipeline([("vec", CountVectorizer(token_pattern=_TOKENS)), ("clf", DummyClassifier(strategy="most_frequent"))])
    if name == "naive_bayes":
        return Pipeline([("vec", CountVectorizer(token_pattern=_TOKENS, stop_words="english", min_df=2)),
                         ("clf", MultinomialNB())])
    if name == "logreg":
        return Pipeline([("vec", TfidfVectorizer(token_pattern=_TOKENS, stop_words="english", min_df=2, sublinear_tf=True)),
                         ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    if name in ("naive_bayes_tuned", "logreg_tuned"):
        # Tuning happens INSIDE each training fold (nested cross-validation), so the reported score is not tuned on its own test data.
        base = build_pipeline(name.removesuffix("_tuned"))
        grid = {"vec__ngram_range": [(1, 1), (1, 2)], "vec__min_df": [2, 3]}
        grid.update({"clf__alpha": [0.1, 0.5, 1.0]} if name.startswith("naive") else {"clf__C": [0.3, 1.0, 3.0, 10.0]})
        return GridSearchCV(base, grid, cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="f1", n_jobs=1)
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


def cross_validate_model(df: pd.DataFrame, name: str, n_splits: int = 5, n_repeats: int = 10, seed: int = 0) -> dict:
    """Mean and spread over repeated stratified k-fold. Metrics for the 'flagged' class are what a selector is judged on."""
    X, y = df["resume_text"].to_numpy(), df["label"].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for tr, te in cv.split(X, y):
        pipe = build_pipeline(name).fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        rows.append((accuracy_score(y[te], pred),
                     precision_score(y[te], pred, zero_division=0),
                     recall_score(y[te], pred, zero_division=0)))
    a = np.array(rows)
    return {"model": name, "n_folds": len(rows),
            "accuracy_mean": float(a[:, 0].mean()), "accuracy_std": float(a[:, 0].std()),
            "precision_flagged_mean": float(a[:, 1].mean()), "precision_flagged_std": float(a[:, 1].std()),
            "recall_flagged_mean": float(a[:, 2].mean()), "recall_flagged_std": float(a[:, 2].std())}


def top_terms(pipe: Pipeline, n: int = 15) -> dict:
    """The words that push a resume towards each class (needs a linear model's coefficients)."""
    terms = np.array(pipe.named_steps["vec"].get_feature_names_out())
    coef = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    return {"flagged": list(terms[order[::-1][:n]]), "not_flagged": list(terms[order[:n]])}


def misclassified(df: pd.DataFrame, name: str, n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """Every resume the model gets wrong when it is held out (each resume is predicted by a model that never saw it)."""
    X, y = df["resume_text"].to_numpy(), df["label"].to_numpy()
    pred = np.empty_like(y)
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y):
        pred[te] = build_pipeline(name).fit(X[tr], y[tr]).predict(X[te])
    bad = np.flatnonzero(pred != y)
    return pd.DataFrame({"index": bad, "true": y[bad], "predicted": pred[bad],
                         "snippet": [X[i][:120].replace("\n", " ") for i in bad]})
