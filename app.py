# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader # Added import
import base64

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Hypertension Risk Predictor", layout="wide", page_icon="🩺")

# --- Configuration ---
MODEL_PATH = 'model.pkl'
ENCODERS_PATH = 'label_encoders.pkl'  
# --- Load Model and Preprocessing Objects ---
@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and label encoders."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file '{MODEL_PATH}' not found")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    try:
        with open(ENCODERS_PATH, 'rb') as f:
            encoders = pickle.load(f)
    except FileNotFoundError:
        st.warning(f"Encoders file '{ENCODERS_PATH}' not found")
        encoders = {}
    except Exception as e:
        st.error(f"Error loading encoders: {e}")
        st.stop()
    return model, encoders

model, encoders = load_model_and_encoders()

# --- SHAP Explainer ---
@st.cache_resource
def load_shap_explainer(_model):
    """Create SHAP explainer without triggering UnhashableParamError."""
    try:
        explainer = shap.TreeExplainer(_model)
        return explainer
    except Exception as e:
        st.error(f"Error initializing SHAP explainer: {e}")
        return None

explainer = load_shap_explainer(_model=model)

# --- PDF Generation ---
def create_pdf_report(input_data, prediction, prediction_proba, shap_fig, suggestions):
    """Create a PDF report of the prediction results."""
    # Save SHAP plot to buffer (as PNG image data)
    shap_buffer = io.BytesIO()
    shap_fig.savefig(shap_buffer, format='png', bbox_inches='tight')
    shap_buffer.seek(0) # Reset buffer pointer to the beginning


    buffer_final = io.BytesIO()
    doc_final = SimpleDocTemplate(buffer_final, pagesize=letter)
    styles = getSampleStyleSheet()
    story_final = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story_final.append(Paragraph("Hypertension Risk Prediction Report", title_style))
    story_final.append(Spacer(1, 20))
    
    # Date
    story_final.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story_final.append(Spacer(1, 20))
    
    # Patient Information
    story_final.append(Paragraph("Patient Information:", styles['Heading2']))
    patient_data = [[key.replace('_', ' ').title(), str(value)] for key, value in input_data.items()]
    patient_table = Table(patient_data)
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story_final.append(patient_table)
    story_final.append(Spacer(1, 20))
    
    # Prediction Results
    story_final.append(Paragraph("Prediction Results:", styles['Heading2']))
    risk_text = "High Risk" if prediction == 1 else "Low Risk"
    risk_color = "red" if prediction == 1 else "green"
    story_final.append(Paragraph(f"Predicted Risk: <font color='{risk_color}'><b>{risk_text}</b></font>", styles['Normal']))
    story_final.append(Paragraph(f"Probability of Low Risk: {prediction_proba[0]:.2%}", styles['Normal']))
    story_final.append(Paragraph(f"Probability of High Risk: {prediction_proba[1]:.2%}", styles['Normal']))
    story_final.append(Spacer(1, 20))
    
    # Treatment Suggestions
    story_final.append(Paragraph("Treatment Suggestions:", styles['Heading2']))
    for suggestion in suggestions:
        story_final.append(Paragraph(suggestion, styles['Normal']))
    story_final.append(Spacer(1, 20))
    
    # SHAP Explanation
    story_final.append(Paragraph("SHAP Explanation:", styles['Heading2']))
    # Add the SHAP plot image using Image from Platypus
    # Create a ReportLab Image object from the BytesIO buffer
    shap_platypus_image = Image(shap_buffer, width=500, height=300) # Adjust width/height as needed
    story_final.append(shap_platypus_image)

    # Build the final PDF
    doc_final.build(story_final)
    buffer_final.seek(0)
    return buffer_final.getvalue() # Return the bytes of the final combined PDF

# --- Streamlit UI ---

st.markdown("""
<style>
    .main-header {
        background-color: #1f77b4;
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-header {
        background-color: #e1f0fa;
        padding: 15px;
        border-radius: 8px;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    .result-card {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .btn-primary {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
    }
    .btn-primary:hover {
        background-color: #0d5a9e;
    }
    .stButton>button {
        background-color: #1f77b4 !important;
        color: white !important;
        border: none !important;
        padding: 10px 20px !important;
        border-radius: 5px !important;
        font-size: 16px !important;
        width: 100% !important;
    }
    .stButton>button:hover {
        background-color: #0d5a9e !important;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header"><h1>🩺 Hypertension Risk Prediction App</h1></div>', unsafe_allow_html=True)
st.markdown("""
This app predicts the risk of hypertension based on patient data and explains the prediction using SHAP.
""")

# --- Input Form ---
st.markdown('<div class="section-header"><h2>Patient Information</h2></div>', unsafe_allow_html=True)
feature_names = [
    'age', 'gender', 'systolic_bp', 'diastolic_bp', 'bmi',
    'glucose_level', 'cholesterol', 'smoker', 'alcohol_use',
    'physical_activity', 'med_adherence'
]

# Create two columns for better layout
col1, col2 = st.columns(2)
input_data = {}

with col1:
    # Numerical Inputs
    input_data['age'] = st.number_input("Age", min_value=0, max_value=120, value=50, step=1)
    input_data['systolic_bp'] = st.number_input("Systolic Blood Pressure (mmHg)", min_value=50.0, max_value=250.0, value=120.0, step=1.0)
    input_data['diastolic_bp'] = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30.0, max_value=150.0, value=80.0, step=1.0)
    input_data['bmi'] = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    input_data['glucose_level'] = st.number_input("Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=100.0, step=1.0)

with col2:
    input_data['cholesterol'] = st.number_input("Cholesterol Level (mg/dL)", min_value=100.0, max_value=400.0, value=200.0, step=1.0)
    # Categorical Inputs
    if 'gender' in encoders:
        gender_options = encoders['gender'].classes_.tolist()
    else:
        gender_options = ['Male', 'Female']
    input_data['gender'] = st.selectbox("Gender", options=gender_options)
    # Binary Inputs (Yes/No -> 1/0)
    input_data['smoker'] = st.radio("Smoker?", options=['No', 'Yes'], index=1)
    input_data['alcohol_use'] = st.radio("Alcohol Use?", options=['No', 'Yes'], index=1)
    input_data['physical_activity'] = st.radio("Regular Physical Activity?", options=['No', 'Yes'], index=0)
    input_data['med_adherence'] = st.radio("Medication Adherence?", options=['No', 'Yes'], index=1)

# Convert binary inputs
input_data['smoker'] = 1 if input_data['smoker'] == 'Yes' else 0
input_data['alcohol_use'] = 1 if input_data['alcohol_use'] == 'Yes' else 0
input_data['physical_activity'] = 1 if input_data['physical_activity'] == 'Yes' else 0
input_data['med_adherence'] = 1 if input_data['med_adherence'] == 'Yes' else 0

# --- Prediction Button ---
st.markdown("<br>", unsafe_allow_html=True)
predict_button = st.button("🔮 Predict Risk", key="predict")

# Initialize session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = None

if predict_button:
    input_df = pd.DataFrame([input_data])
    try:
        input_df = input_df[feature_names]
    except KeyError as e:
        st.error(f"Input data is missing required columns or column names do not match: {e}")
        st.stop()
    input_df_processed = input_df.copy()
    for col, encoder in encoders.items():
        if col in input_df_processed.columns:
            try:
                input_df_processed[col] = encoder.transform(input_df_processed[col])
            except ValueError as e:
                st.error(f"Error encoding '{col}': {e}. Please check the input value.")
                st.stop()
    try:
        prediction = model.predict(input_df_processed)[0]
        prediction_proba = model.predict_proba(input_df_processed)[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()
    # Store results in session state
    st.session_state.results = {
        'input_data': input_data,
        'prediction': prediction,
        'prediction_proba': prediction_proba
    }

# Display results if available
if st.session_state.results:
    results = st.session_state.results
    input_data = results['input_data']
    prediction = results['prediction']
    prediction_proba = results['prediction_proba']

    # Display prediction result
    risk_text = "High Risk" if prediction == 1 else "Low Risk"
    risk_class = "high-risk" if prediction == 1 else "low-risk"
    st.markdown(f'<div class="result-card {risk_class}"><h3>📊 Prediction Result</h3><h4>Predicted Risk: <b>{risk_text}</b></h4><p>Probability of Low Risk: <b>{prediction_proba[0]:.2%}</b></p><p>Probability of High Risk: <b>{prediction_proba[1]:.2%}</b></p></div>', unsafe_allow_html=True)

    # SHAP Explanation
    if explainer:
        st.markdown('<div class="section-header"><h2>🔮 Prediction Explanation (SHAP)</h2></div>', unsafe_allow_html=True)
        st.markdown("This plot shows how each feature contributed to the prediction for this specific patient.")
        try:
            shap_values = explainer.shap_values(input_df_processed)
            shap_values_for_class_1 = shap_values[1] if isinstance(shap_values, list) else shap_values
            base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            feature_values = input_df_processed.iloc[0]
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_for_class_1[0],
                    base_values=base_value,
                    data=feature_values,
                    feature_names=feature_names
                ),
                show=False,
                max_display=10
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            # Store the figure for PDF generation
            st.session_state.results['shap_fig'] = fig
        except Exception as e:
            st.error(f"Error generating SHAP explanation: {e}")
    else:
        st.warning("SHAP explainer could not be initialized. Explanation not available.")

    # Treatment Suggestions
    st.markdown('<div class="section-header"><h2>💊 Treatment Suggestions</h2></div>', unsafe_allow_html=True)
    suggestions = []
    if prediction == 1:
        st.markdown('<div class="result-card high-risk"><h4>🚨 High Risk Patient Detected:</h4>', unsafe_allow_html=True)
        if input_data['systolic_bp'] > 140 or input_data['diastolic_bp'] > 90:
            suggestions.append("- Blood pressure is elevated. Review current antihypertensive regimen.")
        if input_data['bmi'] > 30:
            suggestions.append("- BMI indicates obesity. Recommend weight loss program.")
        if input_data['cholesterol'] > 200:
            suggestions.append("- High cholesterol. Consider lipid-lowering therapy.")
        if input_data['glucose_level'] > 126:
            suggestions.append("- High glucose. Screen for diabetes.")
        if input_data['smoker'] == 1:
            suggestions.append("- Encourage smoking cessation.")
        if input_data['alcohol_use'] == 1:
            suggestions.append("- Advise moderation or cessation of alcohol.")
        if input_data['physical_activity'] == 0:
            suggestions.append("- Recommend regular physical activity.")
        if input_data['med_adherence'] == 0:
            suggestions.append("- Improve medication adherence.")
        suggestions.append("- Schedule regular follow-ups.")
        suggestions.append("- Refer to a specialist if needed.")
        for suggestion in suggestions:
            st.markdown(f"<p>{suggestion}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-card low-risk"><h4>✅ Low Risk Patient:</h4>', unsafe_allow_html=True)
        suggestions = [
            "- Maintain current management plan.",
            "- Encourage healthy lifestyle.",
            "- Continue regular check-ups."
        ]
        for suggestion in suggestions:
            st.markdown(f"<p>{suggestion}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Update session state with suggestions
    st.session_state.results['suggestions'] = suggestions

    # PDF Export Button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📄 Export Results as PDF"):
        if 'shap_fig' in st.session_state.results:
            pdf_buffer = create_pdf_report(
                input_data, 
                prediction, 
                prediction_proba, 
                st.session_state.results['shap_fig'],
                suggestions
            )
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_buffer,
                file_name=f"hypertension_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("SHAP explanation not available for PDF export.")

# Footer
st.markdown("---")
st.markdown("🩺 Hypertension Risk Prediction App | Developed with Streamlit")