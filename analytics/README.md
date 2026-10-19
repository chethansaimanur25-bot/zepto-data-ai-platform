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