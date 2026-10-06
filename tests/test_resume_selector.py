import numpy as np
import pandas as pd
import pytest

import resume_selector as rs


def tiny():
    pos = ["python machine learning model data scientist neural network"] * 12
    neg = ["customer service retail store cashier sales floor"] * 28
    return pd.DataFrame({"resume_text": pos + neg, "label": [1] * 12 + [0] * 28})


def test_load_resumes_reads_the_real_file_with_binary_labels():
    df = rs.load_resumes("resume.csv")
    assert len(df) == 125 and set(df["label"]) == {0, 1}
    assert df["label"].sum() == 33          # 33 flagged, 92 not flagged


def test_no_test_data_reaches_the_vocabulary():
    """The v1 flaw: the vectorizer saw every resume before the split. In a pipeline it is fitted on the training fold only."""
    train = ["alpha beta"] * 10
    pipe = rs.build_pipeline("naive_bayes").fit(train, [0, 1] * 5)
    vocab = pipe.named_steps["vec"].vocabulary_
    assert "alpha" in vocab and "zebra" not in vocab


def test_majority_baseline_scores_the_class_share():
    df = tiny()
    res = rs.cross_validate_model(df, "majority", n_splits=4, n_repeats=2, seed=0)
    assert res["accuracy_mean"] == pytest.approx(28 / 40, abs=0.03)
    assert res["recall_flagged_mean"] == 0.0


def test_a_real_model_beats_the_baseline_on_separable_text():
    df = tiny()
    for name in ("naive_bayes", "logreg"):
        res = rs.cross_validate_model(df, name, n_splits=4, n_repeats=2, seed=0)
        assert res["accuracy_mean"] > 0.95, name


def test_cross_validation_is_seeded_and_reports_spread():
    df = tiny()
    a = rs.cross_validate_model(df, "naive_bayes", n_splits=4, n_repeats=3, seed=7)
    b = rs.cross_validate_model(df, "naive_bayes", n_splits=4, n_repeats=3, seed=7)
    assert a == b
    assert "accuracy_std" in a and a["n_folds"] == 12


def test_top_terms_name_the_words_that_drive_each_class():
    df = tiny()
    pipe = rs.build_pipeline("logreg").fit(df["resume_text"], df["label"])
    terms = rs.top_terms(pipe, n=3)
    assert "python" in terms["flagged"] or "learning" in terms["flagged"]
    assert "customer" in terms["not_flagged"] or "retail" in terms["not_flagged"]


def test_misclassified_lists_each_resume_the_out_of_fold_model_gets_wrong():
    df = tiny()
    df.loc[0, "label"] = 0                   # one mislabelled-looking resume
    wrong = rs.misclassified(df, "logreg", n_splits=4, seed=0)
    assert 0 in set(wrong["index"])
    assert {"index", "true", "predicted", "snippet"} <= set(wrong.columns)


def test_tuned_models_are_nested_and_still_score_well():
    df = tiny()
    for name in ("naive_bayes_tuned", "logreg_tuned"):
        res = rs.cross_validate_model(df, name, n_splits=4, n_repeats=1, seed=0)
        assert res["accuracy_mean"] > 0.95, name
