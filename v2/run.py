"""Regenerates every number and figure in the README: `python v2/run.py` from the repository root."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import resume_selector as rs  # noqa: E402

SEED = 0
df = rs.load_resumes(str(ROOT / "resume.csv"))
share = 1 - df["label"].mean()

# 1. what v1 did, repeated: vocabulary from ALL resumes, one random 75/25 split. Different seeds = different "results".
X_all = CountVectorizer().fit_transform(df["resume_text"])          # v1's leakage: fitted before the split
v1_acc = []
for s in range(200):
    Xtr, Xte, ytr, yte = train_test_split(X_all, df["label"], test_size=0.25, random_state=s)
    v1_acc.append(MultinomialNB().fit(Xtr, ytr).score(Xte, yte))
v1_acc = np.array(v1_acc)

# 2. v2: leak-free pipelines, repeated stratified 5-fold, three models
results = {name: rs.cross_validate_model(df, name, n_splits=5, n_repeats=10, seed=SEED) for name in rs.MODELS}
logreg = rs.build_pipeline("logreg").fit(df["resume_text"], df["label"])
terms = rs.top_terms(logreg, n=12)
wrong = rs.misclassified(df, "logreg", seed=SEED)

out = {"n_resumes": len(df), "n_flagged": int(df["label"].sum()), "majority_share": share,
       "v1_style": {"runs": 200, "accuracy_mean": float(v1_acc.mean()), "accuracy_min": float(v1_acc.min()),
                    "accuracy_max": float(v1_acc.max()), "accuracy_std": float(v1_acc.std())},
       "v2": results, "top_terms": terms,
       "misclassified_logreg": wrong.drop(columns="snippet").to_dict("records")}
(ROOT / "v2" / "results.json").write_text(json.dumps(out, indent=2))

# figure 1: v1's single-split claim vs. what the same method gives across seeds
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.hist(v1_acc, bins=15, color="#4c78a8")
ax.axvline(share, color="#c0392b", ls="--", label=f"always say 'not flagged' ({share:.2f})")
ax.set_xlabel("test accuracy of v1's method, 200 different random splits"); ax.set_ylabel("runs")
ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "v2" / "v1-split-spread.png", dpi=150); plt.close(fig)

# figure 2: v2 comparison with spread
names = list(rs.MODELS); labels = ["always 'not\nflagged'", "Naive Bayes", "Logistic\nregression", "Naive Bayes\n+ tuning", "Logistic reg.\n+ tuning"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, key, title in zip(axes, ("accuracy", "recall_flagged"), ("accuracy", "recall on 'flagged'")):
    m = [results[n][f"{key}_mean"] for n in names]; sd = [results[n][f"{key}_std"] for n in names]
    ax.bar(labels, m, yerr=sd, color=["#999", "#4c78a8", "#59a14f", "#2f5d8c", "#2e7d32"], capsize=4)
    ax.set_title(title); ax.set_ylim(0, 1.05)
fig.suptitle("v2: 5-fold cross-validation, repeated 10 times (bars show the spread)", fontsize=9)
fig.tight_layout(); fig.savefig(ROOT / "v2" / "v2-comparison.png", dpi=150); plt.close(fig)
print(json.dumps({k: out[k] for k in ("v1_style", "v2")}, indent=1)); print(terms); print(len(wrong), "misclassified (logreg)")
