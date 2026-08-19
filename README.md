# Healthcare Readmission Prediction

Predicting 30-day hospital readmissions in diabetes patients using demographic, clinical, and hospital utilization data — built as an end-to-end analytics project covering data cleaning, feature engineering, and comparative model evaluation.

---

## Problem Statement

Hospital readmissions within 30 days are costly and, in the US, trigger Medicare financial penalties under the Hospital Readmissions Reduction Program. This project builds a predictive model to flag diabetes patients at high risk of early readmission at the point of discharge, so care teams can prioritize follow-up and intervention where it matters most.

## Dataset

- **Source:** [Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), UCI Machine Learning Repository
- **Raw size:** 101,766 hospital encounters × 50 features
- **Cleaned size:** 101,763 rows × 48 features (3 rows with an invalid gender label removed; 7 low-value/high-missingness columns dropped; 5 engineered features added)

## Approach

### 1. Data Cleaning
- Replaced placeholder `'?'` values with proper missing-value indicators
- Dropped identifier columns (`encounter_id`, `patient_nbr`) and columns with excessive missingness (`weight` ~97%, `payer_code` ~40%, `medical_specialty` ~49%, plus two unused lab-result columns)
- Removed 3 records with an ambiguous/invalid gender label

### 2. Feature Engineering
- **Diagnosis grouping:** Mapped raw ICD-9 diagnosis codes (`diag_1`, `diag_2`, `diag_3`) into 10 clinically meaningful categories (Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other, Unknown) using official ICD-9 chapter ranges, including symptom-code edge cases (e.g. code 785 grouped with Circulatory)
- **Age:** Converted bracketed age ranges (e.g. `[70-80)`) into numeric midpoints for modeling
- **Target:** Collapsed the original 3-class `readmitted` variable into a binary target — `<30 days = 1`, all else `= 0` — aligned with the clinically actionable 30-day readmission window

### 3. Encoding Strategy
Categorical features were encoded based on their actual structure rather than applying one method uniformly:
- **Binary features** (`gender`, `diabetesMed`, `change`): label-encoded — safe with only 2 classes, since no ordering can be implied either way
- **Nominal features** (`race`, diagnosis categories, `insulin`, `metformin`): One-Hot encoded, since these categories have no natural order and label-encoding them would have introduced a false ranking — a meaningful correction for linear models like Logistic Regression, which would otherwise misread encoded integers as ordinal

### 4. Modeling
Three classifiers were trained and compared on identically encoded features:
- Logistic Regression (interpretable baseline, `class_weight='balanced'`)
- Random Forest (captures non-linear feature interactions)
- XGBoost (`scale_pos_weight` tuned for class imbalance)

## Results

The dataset has a strong class imbalance — only **11.16%** of patients were readmitted within 30 days (7.96:1 ratio). Because of this, model selection prioritized **recall** and **ROC-AUC** over raw accuracy: a model can score well on accuracy just by predicting "not readmitted" for most patients, while missing the exact cases this project exists to catch.

| Model | Recall | ROC-AUC | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.53 | 0.6451 | 0.67 |
| Random Forest | 0.27 | 0.6327 | 0.80 |
| **XGBoost** | **0.57** | **0.6526** | 0.65 |

**XGBoost was selected as the final model.** Random Forest's higher accuracy is a product of the class imbalance, not genuine predictive strength — it correctly identifies low-risk patients but misses the majority of actual readmissions. XGBoost catches the most at-risk patients while maintaining the strongest overall ranking ability between classes.

### Limitations
ROC-AUC across all three models falls in a narrow 0.63–0.65 band, indicating the predictive ceiling here is set by the available features (demographics and hospital utilization history) rather than model choice. Richer clinical data — lab results, medication history detail — would likely yield larger gains than further algorithmic tuning.

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn`

## Repository Structure

```
├── Healthcare_Analytics_CaseStudy.ipynb   # Full analysis notebook
├── diabetic_data_cleaned.csv              # Cleaned dataset (post-processing)
├── visualizations/                        # Saved charts (EDA, model evaluation, dashboard)
├── EXECUTIVE_SUMMARY.txt                  # Generated summary report
└── README.md
```

## Business Impact

The model is intended as a discharge-planning support tool, not an automated decision system: it flags patients for prioritized follow-up (a phone call, closer discharge planning, earlier check-in) rather than making clinical decisions directly. Given that a missed readmission carries a far higher cost — clinically and financially — than an unnecessary follow-up, the model's error trade-off is aligned with that operational reality.

## Future Work

- Reincorporate lab-result features (A1C, glucose serum) with informative-missingness handling, since "test not performed" may itself carry predictive signal
- Hyperparameter tuning via `GridSearchCV`/`Optuna` for further model refinement
- Deploy as an interactive Streamlit application for real-time risk scoring

---

*This project was completed as part of a B.Tech capstone in Production & Industrial Engineering, applying end-to-end data science methodology to a real-world healthcare analytics problem.*
