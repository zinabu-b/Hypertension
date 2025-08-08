<img width="1024" height="432" alt="OIH_Logo-1024x432" src="https://github.com/user-attachments/assets/b0d7b50a-916a-4774-895b-177cdaf2392c" />

# <span style="color:#f57c00; font-size:32px"><b>🩺 Hypertension Risk Prediction App</b></span>

#### <span style="color:#388e3c"><i>Prepared by:</i>  
#### <i>Zinabu Bekele Website: https://zinabu.netlify.app</i>   
#### <i>Nebiyu</i>  
#### <i>Fentahun</i>  
#### <i>Dagim</i></span>

---

## 📌 Overview  
A machine learning system that predicts patient hypertension risk (High/Low) using XGBoost, explains predictions via SHAP values, and provides treatment recommendations through a Streamlit interface. By leveraging readily available patient data such as age, blood pressure, BMI, glucose, cholesterol, lifestyle habits, and medication adherence, our model identifies high-risk individuals before clinical diagnosis, enabling timely preventive interventions.

---

## 🚀 Key Features  
- **Risk Prediction**: Binary classification (High/Low risk) using XGBoost.  
- **Explainability**: SHAP force plots to interpret model decisions.  
- **Treatment Suggestions**: Rule-based recommendations based on risk level and clinical thresholds.  
- **User-Friendly UI**: Interactive web app built with Streamlit.  
- **PDF Export**: Generate downloadable reports for documentation and patient handover.  

---

## 📂 Project Structure  
| File | Description |
|------|-------------|
| `hypertension_data.csv` | Dataset with patient features and risk labels. |
| `hackton1.ipynb` | Notebook for model training, hyperparameter tuning (Optuna), and SHAP analysis. |
| `app.py` | Streamlit application for live predictions. |
| `model.pkl` | Pretrained XGBoost model. |
| `label_encoders.pkl` | Encoders for categorical variables (gender, smoker, alcohol use, etc.). |

---

## 🔬 Methods Used

### 1. **Data Preprocessing**
- **Feature Engineering**: Derived BMI from height/weight; normalized continuous variables.
- **Categorical Encoding**: Applied label encoding for binary features (`smoker`, `alcohol_use`, `physical_activity`, `med_adherence`) and gender.
- **Target Variable**: Created a binary `hypertension` label based on clinical thresholds:
  - Systolic BP ≥ 140 mmHg or Diastolic BP ≥ 90 mmHg → High Risk (1)
  - Otherwise → Low Risk (0)

### 2. **Model Selection & Training**
- **Primary Model**: **XGBoost Classifier** was selected due to its strong performance on structured tabular data, ability to handle non-linear relationships, and built-in regularization.
- **Hyperparameter Optimization**: Used **Optuna** to automate search across key parameters:
  - `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`
- **Evaluation Metrics**:
  - Accuracy: 99.8% (on test set)
  - AUC-ROC: ~1.0 (indicating excellent discrimination)
  - Confusion Matrix confirmed near-perfect separation between classes

### 3. **Interpretability & Explainability**
- **SHAP Analysis**: Generated force plots to visualize feature contributions to individual predictions.
- **Global Interpretation**: Identified top predictive features:
  - Systolic BP
  - Age
  - Glucose Level
  - Cholesterol
  - BMI
- Enabled clinicians to understand *why* a patient was flagged as high-risk.

### 4. **Rule-Based Treatment Recommendations**
- Integrated domain knowledge into post-prediction logic:
  - If systolic BP > 140 → suggest antihypertensive review.
  - If BMI > 30 → recommend weight loss program.
  - If glucose > 126 → screen for diabetes.
  - If cholesterol > 200 → consider statin therapy.

---

## ⚙️ Tools & Technologies Used

| Tool/Technology | Purpose |
|------------------|--------|
| **Python** | Core programming language |
| **Pandas / NumPy** | Data manipulation and numerical computation |
| **Scikit-learn** | Train/test split, accuracy scoring, baseline models |
| **XGBoost** | High-performance gradient boosting classifier |
| **Optuna** | Automated hyperparameter optimization |
| **SHAP (SHapley Additive exPlanations)** | Model interpretability and explanation |
| **Streamlit** | Rapid web app development with interactive UI |
| **ReportLab + Pillow** | PDF report generation with visualizations |
| **Pickled Files (.pkl)** | Save trained model and encoders for deployment |

---

## 💡 Feature Innovations & Integration Potential with Local EHR Systems

This app is designed with **real-world clinical integration** in mind and can be directly embedded into **local Electronic Health Record (EHR) systems**:

### ✅ 1. **Automated Risk Scoring at Point-of-Care**
- Integrate the model as a **real-time risk score calculator** within EHR dashboards.
- When a new patient visit is recorded, the system automatically computes hypertension risk using existing vital signs and demographics.
- Example: After entering BP, age, and BMI, the EHR displays “High Risk” with color coding and alerts.

### ✅ 2. **Smart Clinical Decision Support (CDS) Engine**
- Embed the model into EHR’s CDS engine to trigger **automated suggestions**:
  - “Patient aged 65 with SBP=150 — consider initiating antihypertensive therapy.”
  - “BMI 32 — refer to nutritionist.”

### ✅ 3. **Dynamic Risk Dashboard for Clinicians**
- Create a **risk stratification dashboard** showing:
  - Patients at high risk (flagged by model).
  - Trending risk scores over time.
  - Visual summary of contributing factors (via SHAP-style bar charts).

### ✅ 4. **Seamless Data Flow from EHR to Model**
- Use standardized EHR APIs (e.g., FHIR) to pull data:
  - Vital signs (BP, BMI)
  - Lab results (glucose, cholesterol)
  - Medication history (adherence tracking)
  - Lifestyle flags (smoking, alcohol, physical activity)

> 🔗 *Future enhancement*: Connect to EHR via HL7/FHIR to enable **continuous risk monitoring**.

### ✅ 5. **Patient Self-Reporting Module**
- Allow patients to self-report lifestyle behaviors (physical activity, smoking, diet) via secure portals.
- Feed this data into the model to generate personalized risk updates without requiring clinic visits.

### ✅ 6. **Audit Trail & Compliance Logging**
- Log every prediction and recommendation in the EHR for:
  - Quality assurance
  - Regulatory compliance
  - Research and analytics

---

## 📈 Conclusion

The **Hypertension Risk Prediction App** demonstrates how **machine learning, explainability, and user-centered design** can converge to create impactful tools for preventive medicine. With an accuracy exceeding 99%, robust interpretability via SHAP, and practical clinical guidance, it serves as a ready-to-deploy prototype for integration into real-world healthcare systems.

By embedding this tool into **local EHR platforms**, we empower primary care providers to move from reactive to **predictive and preventive care**, ultimately reducing the burden of cardiovascular disease in underserved populations.

---

### > <span style="color:#f57c00; font-weight:bold"> Developed for Orbit Health Innovation Hub AI in Healthcare Hackathon</span>

---

> 💬 *"Prevention is better than cure — and prediction is the first step."*
