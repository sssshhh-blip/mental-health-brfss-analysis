# 🧠 Mental Health Risk Prediction — BRFSS 2024

**Dataset:** [CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024](https://www.cdc.gov/brfss/index.html)  
**Source:** U.S. Centers for Disease Control and Prevention (CDC)  
**Sample Size:** 457,670 respondents | 301 variables

---

## Project Overview

This project applies supervised machine learning methods to the 2024 CDC BRFSS dataset to identify key behavioral, socioeconomic, and health-related predictors of mental health distress among U.S. adults.

The analysis is structured around two modeling scenarios:

- **Scenario A (Classification):** Predicting whether an individual experiences *frequent mental distress* — defined as 14 or more days of poor mental health in the past 30 days — using Logistic Regression, Linear/Quadratic Discriminant Analysis (LDA/QDA), and Random Forest Classifier.
- **Scenario B (Regression):** Predicting the *continuous number of poor mental health days* (0–30) using Linear Regression and Random Forest Regressor.

---

## Methodology

**1. Feature Selection**  
37 clinically and socially relevant variables were selected from the original 301-column dataset, covering dimensions such as physical health, healthcare access, socioeconomic status, disability, lifestyle behaviors, and demographics.

**2. Data Cleaning & Preprocessing**  
- Recoded survey-specific response codes (e.g., "Don't know" = 77, "Refused" = 99) as missing values  
- Imputed missing values using median (continuous variables) or mode (categorical variables)  
- Dropped variables with more than 200,000 missing observations  
- Standardized binary variables to 0/1 encoding with a consistent baseline category  

**3. Exploratory Data Analysis (EDA)**  
- Correlation heatmaps, distribution plots, and interaction point plots  
- Identified variables with significant mean differences in mental health outcomes across groups  

**4. Modeling**  
- Applied cross-validation (StratifiedKFold for classification, KFold for regression)  
- Tuned Random Forest hyperparameters via GridSearchCV  
- KNN was considered but excluded due to computational constraints and the curse of dimensionality (~27 features)  
- Tested PCA + QDA as an iteration; reverted to original model after performance declined  

**5. Evaluation**  
- Scenario A: Precision-Recall curve and Average Precision Score (chosen over accuracy due to class imbalance)  
- Scenario B: R² and RMSE  

---

## Key Findings

| Model | Scenario | Performance |
|---|---|---|
| QDA | A (Classification) | Recall (class 1) = 0.586 |
| Random Forest | A (Classification) | Average Precision = 0.62 |
| Linear Regression | B (Regression) | R² ≈ 0.10 |
| Random Forest | B (Regression) | R² = 0.327, RMSE = 6.84 |

**Top predictors of mental health distress (Random Forest feature importance):**
1. `ADDEPEV3` — Ever diagnosed with depression (~28% importance)
2. `DECIDE` — Difficulty concentrating or making decisions (~18%)
3. `PHYSHLTH` — Number of days physical health was poor (~15%)

---

## Tools & Libraries

`Python` · `pandas` · `scikit-learn` · `statsmodels` · `matplotlib` · `seaborn` · `numpy`

---

## Data Source

Data was obtained from the official CDC BRFSS website. Due to file size, the raw dataset is not included in this repository.  
Download here: [https://www.cdc.gov/brfss/annual_data/annual_2024.html](https://www.cdc.gov/brfss/annual_data/annual_2024.html)
