# Analytics module (Titanic)

Run order: `python analytics/01_eda.py` (Part A). Part B scripts follow below as they are added.
The raw dataset is loaded from the network exactly once (`sns.load_dataset`) and saved as `titanic.csv`; every later step works from that CSV.

## Part A: profiling, cleaning, data story

### Profiling
Shape (891, 15). Missing values measured: deck 77.22%, age 19.87%, embarked 0.22%, embark_town 0.22%.

### Missing-value handling (threshold rule)
| Column | Missing | Rule | Action |
|---|---|---|---|
| deck | 77.22% | over 30% | Dropped the column: imputing three quarters of a column would invent most of its values. |
| age | 19.87% | 5-30% | Median imputation (median is robust to the age outliers below). |
| embarked | 0.22% | under 5% | Dropped the rows. |
| embark_town | 0.22% | under 5% | Dropped the rows (same 2 passengers). |

Cleaned shape (889, 14), 0 missing cells. Caveat: roughly 177 ages are imputed at the median (28), so they all fall in the 19-35 band.

### Univariate analysis
- age: Q1 22.00, Q3 35.00, IQR 13.00, bounds [2.50, 54.50], **65 outliers**.
- fare: Q1 7.90, Q3 31.00, IQR 23.10, bounds [-26.76, 65.66], **114 outliers**.
- fare mean 32.10 > median 14.45 > mode 8.05, so fare is **right-skewed**: a small number of very expensive tickets pull the mean far above the median.

### Bivariate analysis
- Survival by sex: female 74.04%, male 18.89%.
- Survival by class: class 1 62.62% (n=214), class 2 47.28% (n=184), class 3 24.24% (n=491).
- Survival by sex and class: female 96.74% / 92.11% / 50.00% (classes 1/2/3); male 36.89% / 15.74% / 13.54%.
- Correlation matrix on survived, pclass, age, sibsp, parch, fare (adult_male and alone excluded as derived flags); heatmap saved as `charts/corr_heatmap.png`.
- Strongest correlation: **pclass vs fare, r = -0.548**. A higher class number (cheaper ticket) goes with a lower fare.
- Second strongest: **sibsp vs parch, r = 0.415**. Passengers with a spouse or siblings aboard often also had parents or children aboard, i.e. family groups.

### Data story (5 charts)
1. `story1_sex.png`: survival by sex. Women survived at 74.04% and men at 18.89%, a gap of about 55 points. Sex is the single strongest separator of outcomes, consistent with a women-first evacuation.
2. `story2_class_sex.png`: survival by class and sex. Women in classes 1 and 2 survived at 96.74% and 92.11%, but women in class 3 only at 50.00%. Men fell from 36.89% in class 1 to 13.54% in class 3. Class mattered inside each sex, and third-class women (50.00%) still did better than first-class men (36.89%).
3. `story3_age.png`: age by survival. The median age is 28 for both survivors and non-survivors, and the correlation of age with survival is only -0.070. Age alone barely separates outcomes (imputation at the median also pulls both groups toward 28).
4. `story4_fare_class.png`: fare by class and survival, log scale because fare is right-skewed. Median fares are 58.69, 14.25 and 8.05 for classes 1, 2 and 3. Fare correlates with survival (r = 0.255) mainly because it is a proxy for class (pclass vs fare r = -0.548), not as an independent effect.
5. `story5_age_sex_heatmap.png`: survival by age group and sex. Boys under 13 survived at 56.8%, against 16.3% for men aged 19-35, while girls survived at 59.4%, so the sex gap almost vanishes among children. Women aged 13-60 survived at 75-77%. The 100% for women over 60 rests on a very small group and should not be over-read.

Overall: sex was the strongest factor, class the second, and age mattered mainly for children.

### Exploratory standardization (EDA sanity check only, not used by the model)
| | age mean | age std | fare mean | fare std |
|---|---|---|---|---|
| before | 29.315 | 12.978 | 32.097 | 49.67 |
| after (z-score) | 0.0 | 1.0 | 0.0 | 1.0 |

## Part B: predictive modelling

Run order: `python analytics/02_modeling.py`, then `03_regression.py`, then `04_save_pipeline.py`. All three read the committed `titanic.csv`; none of them calls `sns.load_dataset` again.

### Design decisions
- **Target and features:** target `survived`; features pclass, sex, age, sibsp, parch, fare, embarked. I excluded `alive` (it is the target in another form, so using it would leak the answer) and the redundant columns who, adult_male, alone, class, embark_town and deck.
- **Stratified split (80/20, random_state=42), done before any preprocessing.** Class balance is 61.62% died / 38.38% survived. Stratification keeps that ratio in both splits (train 0.3834, test 0.3855), so test metrics are not distorted by an unlucky split of the minority class.
- **Preprocessing inside a Pipeline/ColumnTransformer, fit on the training split only:** numeric columns get median imputation then StandardScaler; sex and embarked get most-frequent imputation then one-hot encoding. The pipeline imputes missing values itself because Part B reads the raw CSV.
- **Train 712 rows, test 179 rows.**

### Classifier comparison (test set)
| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8437 |
| Decision Tree (max_depth=4) | 0.7933 | 0.8636 | 0.5507 | 0.6726 | 0.8292 |
| Random Forest (200 trees) | 0.8156 | 0.8000 | 0.6957 | 0.7442 | 0.8300 |

Confusion matrices as [[TN, FP], [FN, TP]]: Logistic Regression [[98, 12], [23, 46]]; Decision Tree [[104, 6], [31, 38]]; Random Forest [[98, 12], [21, 48]]. Plots: `charts/confusion_matrices.png`, `charts/roc_curves.png`, and the labelled tree in `charts/decision_tree.png`.

### Imbalance handling (Logistic Regression, train counts 439 died / 273 survived)
| Variant | Precision | Recall | F1 |
|---|---|---|---|
| Baseline | 0.7931 | 0.6667 | 0.7244 |
| class_weight='balanced' | 0.7297 | 0.7826 | 0.7552 |
| SMOTE (training data only, inside the pipeline) | 0.7397 | 0.7826 | 0.7606 |

Both fixes raised recall by about 11.6 points at a cost of roughly 5 points of precision, so F1 improved by 3-4 points. SMOTE finished slightly ahead (0.7606 vs 0.7552), but that gap is about one prediction on 179 test rows, so the two are effectively tied. The imbalance is mild (38/62), which is why the gains are modest.

### Hyperparameter tuning (Random Forest, oob_score=True)
GridSearchCV, 5-fold, scoring F1, over n_estimators [100, 200, 300], max_depth [3, 5, 7, None] and max_features ['sqrt', 'log2', 0.5].
- Best parameters: max_depth=3, max_features=0.5, n_estimators=100.
- Best cross-validated F1: 0.7469. OOB score of the best model: 0.823.
- Tuned model on the test set: accuracy 0.8045, precision 0.8036, recall 0.6522, F1 0.72, AUC 0.8375.

The tuned forest did not beat the default forest on the test set (F1 0.72 vs 0.7442). Tuning optimised cross-validated F1 on the training data, and 179 test rows is a small sample. The OOB score is an accuracy estimate on training rows left out of each tree and is not directly comparable to test accuracy.

### Regression side-task: predict fare
Predictors: pclass, age, sibsp, parch, sex, embarked (survived excluded because it is an outcome known only after the voyage). Linear regression, same train-only preprocessing.

| MAE | RMSE | R2 | Adjusted R2 |
|---|---|---|---|
| 20.8094 | 30.4731 | 0.3999 | 0.3679 |

**Heteroscedasticity: yes.** The Spearman correlation between the absolute residual and the predicted fare is 0.573 (p < 0.001), and the residual plot (`charts/regression_residuals.png`) fans out as predicted fare grows. This is expected because fare is strongly right-skewed. A log-transformed target would likely reduce it; I did not refit.

### Model comparison (two separate metric groups, not on a shared scale)
Classification (test set):

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8437 |
| Decision Tree | 0.7933 | 0.8636 | 0.5507 | 0.6726 | 0.8292 |
| Random Forest | 0.8156 | 0.8000 | 0.6957 | 0.7442 | 0.8300 |

Regression (fare, test set):

| Model | MAE | RMSE | R2 | Adjusted R2 |
|---|---|---|---|---|
| Linear Regression | 20.8094 | 30.4731 | 0.3999 | 0.3679 |

### Recommendation
I would deploy the Random Forest. It has the best accuracy (0.8156), recall (0.6957) and F1 (0.7442) of the three classifiers, and recall matters here because we want to find as many survivors as possible. The Decision Tree has the highest precision (0.8636) but misses 31 of the 69 survivors (recall 0.5507). Logistic Regression has the best AUC (0.8437) and is the better choice if ranking quality matters more than the 0.5 cut-off. Tuning did not improve the forest on held-out data. The differences are small and come from a single 179-row test split, so they should be confirmed with cross-validation before production use.

### Saved pipeline
`best_pipeline.joblib` holds the complete fitted pipeline (imputers, encoder, scaler and the Random Forest) as one object, saved with `joblib.dump`. `04_save_pipeline.py` regenerates it, reloads it with `joblib.load` and predicts on raw, unpreprocessed rows, including a row with missing age and embarked.