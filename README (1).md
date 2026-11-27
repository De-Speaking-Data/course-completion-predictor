# 🎓 HarvardX Course Completion Predictor

A machine learning app that estimates whether an online-course student will earn a certificate, based on their engagement with the course.

**🔗 Live demo:** https://course-completion-predictor-gi9xkr9muggpcvr7b4cukk.streamlit.app/
📓 **Notebook:** [Course_Completion_Analysis.ipynb](Course_Completion_Analysis.ipynb)

---

## Problem

Only about **1.8%** of students in HarvardX's 2012–13 online courses earned a certificate. Spotting likely completers (and the students who need support) is hard because the data is heavily imbalanced: a model that always predicts "fail" would be 98% accurate and completely useless.

## Data

- Online Course User Engagement Data (HarvardX person-course records, 2012–13): 20,582 enrollments, 5 courses
- Target: `passed` (derived from `certified`), 365 positives (1.77%)
- Features: course, country, gender, education level, year of birth, events, active days, videos played, chapters accessed, forum posts, active duration

## Approach

1. **Cleaning:** removed an empty column, imputed missing engagement counts with 0, handled one row with missing timestamps.
2. **Feature engineering:** `age` from year of birth, `active_duration_days` from first to last activity.
3. **Leakage control:** dropped `grade`, `certified`, and ID/index columns. Checked `explored` and confirmed the results do not depend on it.
4. **Encoding:** countries under 1% grouped as "Other", then one-hot encoding.
5. **Imbalance handling:** class weights (compared against SMOTE).
6. **Evaluation:** stratified 80/20 split plus 5-fold cross-validation, using PR-AUC because accuracy is misleading at this imbalance.
7. **Threshold tuning:** decision threshold chosen from cross-validated predictions on the training set only, not the test set.

## Results

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|
| Random Forest (balanced) | 0.654 | 0.741 | 0.995 |
| **HistGradientBoosting (balanced)** | **0.699** | **0.780** | 0.983 |

**Final model:** HistGradientBoosting, threshold = 0.88

| Test metric (class "passed") | Value |
|---|---|
| Precision | 0.62 |
| Recall | 0.89 |
| F1 | 0.73 |

The model catches about 89% of students who go on to earn a certificate, while roughly 38% of the students it flags do not pass.

## Limitations

- **Not early prediction.** Engagement features are totals over the whole course, so the model identifies likely completers after the fact. A real early-warning system would use only the first few weeks of activity.
- **Small positive class.** The test set has only 73 passers, so test metrics are noisy (roughly ±5 points).
- **Sparse demographics.** About 94% of gender, education and birth-year values were missing, so these add little.
- Trained on 2012–13 data; patterns may not transfer to current courses.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | Web app interface |
| `model.joblib` | Trained HistGradientBoosting model |
| `meta.json` | Feature columns, category lists and decision threshold |
| `requirements.txt` | Python dependencies |

## Author

**Elijah** · Data science · [Speaking Data](https://www.linkedin.com/)
