# Resume selector v2: plan

## What v1 was

A Coursera guided project: Naive Bayes on bag-of-words counts, one random split (no seed), accuracy 1.00 and 0.97 on test sets of 32 and 38 resumes. Preserved as the git tag `v1-coursework`.

## What is wrong with v1's evidence

1. **The vocabulary was learned from all 125 resumes before the split**, so the test resumes shaped the features (mild leakage for a count vectorizer, but it is leakage).
2. **One unseeded split of 32 to 38 resumes**: one resume is 3 percentage points, and every run gives different numbers.
3. **No baseline.** 74% of the resumes are `not_flagged`, so predicting "not flagged" for everyone scores 0.74. 0.97 means little without that line.
4. **No error analysis**: nothing says which resumes are missed or what the model learned.

## v2 goals

- Fit every preprocessing step inside the cross-validation fold (a scikit-learn `Pipeline`), so no test data touches training.
- Report mean and spread over repeated stratified 5-fold cross-validation, seeded, instead of one split.
- Compare against a majority-class baseline and against at least one stronger model (TF-IDF + logistic regression), so the Naive Bayes number has context.
- Report the metric that matters for a selector: recall and precision for the `flagged` class, not accuracy alone.
- Error analysis: the most informative terms per class, and the resumes the model gets wrong.

## Non-goals

No new data (125 labelled resumes only), no deep models, no claim beyond what 125 resumes can support. The README will say the sample is small.

## Layout

- `resume_selector.py`: loading, pipelines, evaluation (importable, tested)
- `tests/test_resume_selector.py`
- `v2/run.py`: runs the experiment, writes `v2/results.json` and the figures
- README: "Version history" table linking `v1-coursework` and v2

## Success criteria

The README states v1's headline number, v2's cross-validated numbers with their spread, and the baseline, side by side, and every number in it can be regenerated with `python v2/run.py`.
