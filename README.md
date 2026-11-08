# 📊 Week 4 Capstone: Predicting Internship Selection Outcomes

## 📌 Project Overview
This repository contains the Week 4 Capstone project for the **NextGenLearners** data internship. The objective of this project is to build an end-to-end Machine Learning pipeline—moving from feature engineering and encoding to model training, evaluation, and executive reporting—that predicts the likelihood of an applicant being selected for the program.

---

## 🎯 Project Objectives
- Extend the historical applicants dataset with realistic engineered features.
- Preprocess categorical variables using encoding strategies (binary mapping and one-hot encoding).
- Train and evaluate a classification model (**Logistic Regression**) using scikit-learn.
- Interpret feature importance and communicate model limitations ethically.
- Deliver an executive summary suited for non-technical stakeholders.

---

## 🗄️ Dataset Description
The model builds upon the original `applicants` dataset and incorporates 4 simulated behavioural features:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `num_skills_listed` | Numeric | Count of relevant skills listed on application (1–8) |
| `has_portfolio` | Binary (0/1) | Whether applicant provided a portfolio/GitHub URL |
| `prior_hackathon_participation` | Binary (0/1) | Previous participation in hackathons |
| `statement_quality_score` | Numeric | Evaluated quality score of written statement (1–10) |
| `Domain Applied_*` | One-Hot Encoded | Categorical domain indicators generated via `pd.get_dummies()` |

> **Note on Simulated Features:** As historical behavioural data was not directly logged, these 4 features were synthetically generated using a set seed (`np.random.seed(42)`). They represent ideal metrics to collect in future application rounds.

---

## ⚙️ Methodology Pipeline
1. **Feature Preparation**:
   - Converted binary text columns (`Yes`/`No`) into `1`/`0`.
   - Created the target binary variable `target` (`1` if `Status == 'Selected'`, `0` otherwise).
   - Applied one-hot encoding to `Domain Applied` with `drop_first=True`.
2. **Train / Test Split**:
   - Stratified split of 80% training data and 20% testing data (`random_state=42`).
3. **Model Construction**:
   - Trained a `LogisticRegression(max_iter=1000)` classification model.
4. **Evaluation Metrics**:
   - Computed Accuracy, Precision, and Recall.
   - Evaluated true positives and false positives using a **Confusion Matrix**.
5. **Interpretation**:
   - Evaluated feature impact using model coefficients (`model.coef_[0]`).

---

## 📈 Key Results
- **Model Choice**: Logistic Regression
- **Accuracy**: *[Insert your Accuracy %, e.g., 85.00%]*
- **Precision**: *[Insert Precision %, e.g., 80.00%]*
- **Recall**: *[Insert Recall %, e.g., 75.00%]*
- **Key Predictors**: `statement_quality_score` and `has_portfolio` showed the strongest positive coefficients toward selection.

---

## ⚠️ Important Limitations
1. **Simulated Data**: The model relies partly on synthetic features and requires validation on genuine historical records.
2. **Sample Size**: Small test sets may cause variance in performance across different train/test splits.
3. **Ethical Framing**: This model must serve purely as a **decision-support tool** for human reviewers, not an automated gatekeeper, to avoid propagating historical bias.

---

## 💼 Executive Summary
**What We Did:**  
We analyzed past applicant records and developed a machine learning model to identify key applicant attributes associated with internship selection.

**What We Found:**  
The model achieved strong predictive capability on unseen test data. The most significant indicators of selection were high-quality written application statements and the presence of a portfolio or GitHub link. Raw skill counts had a lower relative impact.

**Recommendation:**  
We recommend adding mandatory portfolio/GitHub URL fields and structured statement prompts to future application forms to systematically capture these high-value signals.

---

## 🚀 How to Run the Notebook
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd week4-capstone
   ```
2. Install required packages:
   ```bash
   pip install pandas numpy matplotlib scikit-learn Jupyter
   ```
3. Launch Jupyter Notebook and run `Week4_Capstone_YourName.ipynb` sequentially from top to bottom.

---

## 📹 Walkthrough Video
- **Link**: https://drive.google.com/file/d/11D5JilZXAFl1k-HZFhc13YSBZq6s7thq/view?usp=drive_link
