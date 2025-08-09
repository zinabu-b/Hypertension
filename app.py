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
from reportlab.lib.utils import ImageReader
import base64
import warnings
from typing import Dict, Any, Tuple, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Hypertension Risk Predictor", 
    layout="wide", 
    page_icon="🩺",
    initial_sidebar_state="collapsed"
)

# --- Configuration ---
MODEL_PATH = 'model.pkl'
ENCODERS_PATH = 'label_encoders.pkl'

# Feature configuration - Order matters! Must match model training order
FEATURE_CONFIG = {
    'age': {'min': 0, 'max': 120, 'default': 50, 'step': 1, 'type': 'numeric'},
    'systolic_bp': {'min': 50.0, 'max': 250.0, 'default': 120.0, 'step': 1.0, 'type': 'numeric'},
    'diastolic_bp': {'min': 30.0, 'max': 150.0, 'default': 80.0, 'step': 1.0, 'type': 'numeric'},
    'bmi': {'min': 10.0, 'max': 50.0, 'default': 25.0, 'step': 0.1, 'type': 'numeric'},
    'glucose_level': {'min': 50.0, 'max': 300.0, 'default': 100.0, 'step': 1.0, 'type': 'numeric'},
    'cholesterol': {'min': 100.0, 'max': 400.0, 'default': 200.0, 'step': 1.0, 'type': 'numeric'},
    'gender': {'options': ['Male', 'Female'], 'type': 'categorical'},
    'smoker': {'options': ['No', 'Yes'], 'type': 'binary'},
    'alcohol_use': {'options': ['No', 'Yes'], 'type': 'binary'},
    'physical_activity': {'options': ['No', 'Yes'], 'type': 'binary'},
    'med_adherence': {'options': ['No', 'Yes'], 'type': 'binary'}
}

# This will be dynamically determined from the model
FEATURE_NAMES = None

# Risk thresholds for clinical guidelines
RISK_THRESHOLDS = {
    'systolic_bp_high': 140,
    'diastolic_bp_high': 90,
    'bmi_obese': 30,
    'cholesterol_high': 200,
    'glucose_high': 126
}

# --- Utility Functions ---
def validate_input_data(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize input data."""
    validated_data = {}
    
    for feature, value in input_data.items():
        if feature in FEATURE_CONFIG:
            config = FEATURE_CONFIG[feature]
            
            if config['type'] == 'numeric':
                # Ensure numeric values are within bounds
                validated_data[feature] = max(config['min'], min(config['max'], value))
            elif config['type'] == 'binary':
                # Convert binary to 0/1
                validated_data[feature] = 1 if value == 'Yes' else 0
            else:
                validated_data[feature] = value
        else:
            validated_data[feature] = value
    
    return validated_data

def get_clinical_alerts(input_data: Dict[str, Any]) -> List[str]:
    """Generate clinical alerts based on input values."""
    alerts = []
    
    # Critical BP values
    if input_data['systolic_bp'] >= 180 or input_data['diastolic_bp'] >= 110:
        alerts.append("⚠️ CRITICAL: Blood pressure values suggest hypertensive crisis - immediate medical attention required")
    elif input_data['systolic_bp'] >= 160 or input_data['diastolic_bp'] >= 100:
        alerts.append("⚠️ WARNING: Stage 2 hypertension detected")
    elif input_data['systolic_bp'] >= 140 or input_data['diastolic_bp'] >= 90:
        alerts.append("⚠️ ALERT: Stage 1 hypertension detected")
    
    # Other risk factors
    if input_data['glucose_level'] >= 200:
        alerts.append("⚠️ CRITICAL: Very high glucose levels - diabetes screening urgent")
    elif input_data['glucose_level'] >= 126:
        alerts.append("⚠️ ALERT: Elevated glucose - diabetes screening recommended")
    
    if input_data['bmi'] >= 35:
        alerts.append("⚠️ WARNING: Severe obesity (BMI ≥35)")
    elif input_data['bmi'] >= 30:
        alerts.append("⚠️ ALERT: Obesity detected (BMI ≥30)")
    
    return alerts

# --- Load Model and Preprocessing Objects ---
@st.cache_resource
def load_model_and_encoders() -> Tuple[Any, Dict[str, Any], List[str]]:
    """Load the trained model and label encoders with better error handling."""
    global FEATURE_NAMES
    
    model = None
    encoders = {}
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        
        # Get feature names from the model if available
        if hasattr(model, 'feature_names_in_'):
            FEATURE_NAMES = model.feature_names_in_.tolist()
            logger.info(f"Feature names from model: {FEATURE_NAMES}")
        elif hasattr(model, 'feature_names_'):
            FEATURE_NAMES = model.feature_names_
            logger.info(f"Feature names from model: {FEATURE_NAMES}")
        else:
            # Fallback to default order based on common model training patterns
            FEATURE_NAMES = ['age', 'systolic_bp', 'diastolic_bp', 'bmi', 'glucose_level', 'cholesterol', 
                           'gender', 'smoker', 'alcohol_use', 'physical_activity', 'med_adherence']
            logger.warning(f"Model doesn't have feature names, using default: {FEATURE_NAMES}")
            
    except FileNotFoundError:
        st.error(f"❌ Model file '{MODEL_PATH}' not found. Please ensure the model file is in the correct location.")
        st.info("💡 Place your trained model file in the same directory as this app.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        logger.error(f"Error loading model: {e}")
        st.stop()
    
    try:
        with open(ENCODERS_PATH, 'rb') as f:
            encoders = pickle.load(f)
        logger.info("Encoders loaded successfully")
    except FileNotFoundError:
        st.warning(f"⚠️ Encoders file '{ENCODERS_PATH}' not found. Using default encoders.")
        # Create default encoders if file not found
        encoders = {'gender': type('MockEncoder', (), {'classes_': np.array(['Male', 'Female'])})()}
    except Exception as e:
        st.error(f"❌ Error loading encoders: {e}")
        logger.error(f"Error loading encoders: {e}")
        st.stop()
    
    return model, encoders, FEATURE_NAMES

# --- SHAP Explainer ---
@st.cache_resource
def load_shap_explainer(_model) -> Optional[Any]:
    """Create SHAP explainer with better error handling."""
    try:
        explainer = shap.TreeExplainer(_model)
        logger.info("SHAP explainer initialized successfully")
        return explainer
    except Exception as e:
        logger.error(f"Error initializing SHAP explainer: {e}")
        st.warning("⚠️ SHAP explainer could not be initialized. Predictions will work but explanations won't be available.")
        return None

# --- PDF Generation ---
def create_pdf_report(input_data: Dict[str, Any], prediction: int, prediction_proba: np.ndarray, 
                     shap_fig: Optional[plt.Figure], suggestions: List[str], alerts: List[str], 
                     feature_names: List[str]) -> bytes:
    """Create a comprehensive PDF report with improved formatting."""
    buffer_final = io.BytesIO()
    doc_final = SimpleDocTemplate(buffer_final, pagesize=letter, topMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story_final = []

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,
        textColor=colors.darkblue
    )
    
    alert_style = ParagraphStyle(
        'AlertStyle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.red,
        leftIndent=20
    )

    # Title and header
    story_final.append(Paragraph("🩺 Hypertension Risk Prediction Report", title_style))
    story_final.append(Spacer(1, 20))
    
    # Date and time
    story_final.append(Paragraph(f"<b>Generated on:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story_final.append(Spacer(1, 20))
    
    # Clinical alerts if any
    if alerts:
        story_final.append(Paragraph("<b>🚨 Clinical Alerts:</b>", styles['Heading2']))
        for alert in alerts:
            story_final.append(Paragraph(alert, alert_style))
        story_final.append(Spacer(1, 20))
    
    # Patient Information
    story_final.append(Paragraph("<b>👤 Patient Information:</b>", styles['Heading2']))
    patient_data = []
    for key, value in input_data.items():
        display_key = key.replace('_', ' ').title()
        if key in ['systolic_bp', 'diastolic_bp']:
            display_value = f"{value} mmHg"
        elif key == 'glucose_level':
            display_value = f"{value} mg/dL"
        elif key == 'cholesterol':
            display_value = f"{value} mg/dL"
        elif key == 'bmi':
            display_value = f"{value:.1f} kg/m²"
        elif isinstance(value, (int, float)) and key not in ['smoker', 'alcohol_use', 'physical_activity', 'med_adherence']:
            display_value = str(value)
        else:
            display_value = 'Yes' if value == 1 else 'No' if value == 0 else str(value)
        
        patient_data.append([display_key, display_value])
    
    patient_table = Table(patient_data, colWidths=[2.5*inch, 2*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkblue),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    story_final.append(patient_table)
    story_final.append(Spacer(1, 20))
    
    # Prediction Results
    story_final.append(Paragraph("<b>📊 Prediction Results:</b>", styles['Heading2']))
    risk_text = "High Risk" if prediction == 1 else "Low Risk"
    risk_color = colors.red if prediction == 1 else colors.green
    
    story_final.append(Paragraph(f"<b>Predicted Risk:</b> <font color='{risk_color}'><b>{risk_text}</b></font>", styles['Normal']))
    story_final.append(Paragraph(f"<b>Low Risk Probability:</b> {prediction_proba[0]:.1%}", styles['Normal']))
    story_final.append(Paragraph(f"<b>High Risk Probability:</b> {prediction_proba[1]:.1%}", styles['Normal']))
    story_final.append(Spacer(1, 20))
    
    # Treatment Suggestions
    story_final.append(Paragraph("<b>💊 Clinical Recommendations:</b>", styles['Heading2']))
    for suggestion in suggestions:
        story_final.append(Paragraph(suggestion, styles['Normal']))
    story_final.append(Spacer(1, 20))
    
    # SHAP Explanation
    if shap_fig is not None:
        story_final.append(Paragraph("<b>🔍 Model Explanation (SHAP):</b>", styles['Heading2']))
        story_final.append(Paragraph("This chart shows how each patient factor contributed to the risk prediction:", styles['Normal']))
        
        # Save SHAP plot to buffer
        shap_buffer = io.BytesIO()
        shap_fig.savefig(shap_buffer, format='png', bbox_inches='tight', dpi=150)
        shap_buffer.seek(0)
        
        shap_image = Image(shap_buffer, width=6*inch, height=3.5*inch)
        story_final.append(shap_image)
    
    # Disclaimer
    story_final.append(Spacer(1, 30))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey,
        alignment=1
    )
    story_final.append(Paragraph(
        "<b>Disclaimer:</b> This prediction is for clinical decision support only and should not replace professional medical judgment. "
        "Always consult with healthcare professionals for medical decisions.",
        disclaimer_style
    ))

    # Build PDF
    doc_final.build(story_final)
    buffer_final.seek(0)
    return buffer_final.getvalue()

# --- Enhanced Treatment Suggestions ---
def get_treatment_suggestions(input_data: Dict[str, Any], prediction: int, feature_names: List[str]) -> List[str]:
    """Generate comprehensive treatment suggestions based on risk factors."""
    suggestions = []
    
    if prediction == 1:  # High Risk
        # Priority interventions
        if input_data['systolic_bp'] > RISK_THRESHOLDS['systolic_bp_high'] or input_data['diastolic_bp'] > RISK_THRESHOLDS['diastolic_bp_high']:
            if input_data['systolic_bp'] >= 160 or input_data['diastolic_bp'] >= 100:
                suggestions.append("🔴 <b>URGENT:</b> Initiate or intensify antihypertensive therapy immediately (Stage 2 HTN)")
            else:
                suggestions.append("🔴 <b>HIGH PRIORITY:</b> Review and optimize antihypertensive regimen (Stage 1 HTN)")
        
        # Lifestyle modifications
        if input_data['bmi'] > RISK_THRESHOLDS['bmi_obese']:
            weight_loss_target = max(25, input_data['bmi'] - 5)
            suggestions.append(f"🟡 <b>LIFESTYLE:</b> Weight reduction program - target BMI reduction to {weight_loss_target:.1f} (5-10% initial weight loss)")
        
        if input_data['cholesterol'] > RISK_THRESHOLDS['cholesterol_high']:
            suggestions.append("🟡 <b>MEDICATION:</b> Consider statin therapy for cholesterol management")
        
        if input_data['glucose_level'] > RISK_THRESHOLDS['glucose_high']:
            if input_data['glucose_level'] >= 200:
                suggestions.append("🔴 <b>URGENT:</b> Comprehensive diabetes evaluation and immediate glucose management")
            else:
                suggestions.append("🟡 <b>SCREENING:</b> Diabetes screening and glucose management consultation")
        
        # Risk factor modifications
        if input_data['smoker'] == 1:
            suggestions.append("🟡 <b>CESSATION:</b> Smoking cessation program - refer to tobacco cessation specialist")
        
        if input_data['alcohol_use'] == 1:
            suggestions.append("🟡 <b>LIFESTYLE:</b> Alcohol counseling - limit to ≤2 drinks/day (men) or ≤1 drink/day (women)")
        
        if input_data['physical_activity'] == 0:
            suggestions.append("🟢 <b>LIFESTYLE:</b> Structured exercise program - 150 min/week moderate aerobic activity")
        
        if input_data['med_adherence'] == 0:
            suggestions.append("🔴 <b>HIGH PRIORITY:</b> Medication adherence counseling and monitoring system")
        
        # Follow-up
        suggestions.append("🔵 <b>MONITORING:</b> Follow-up in 2-4 weeks, then monthly until BP controlled")
        suggestions.append("🔵 <b>REFERRAL:</b> Consider cardiology/nephrology referral if resistant hypertension")
        
    else:  # Low Risk
        suggestions.append(" 🟢 Maintain current management plan.")
        suggestions.append(" 🟢 Encourage healthy lifestyle.")
        suggestions.append(" 🟢 Continue regular check-ups.")
        
        # # Preventive recommendations
        # if input_data['bmi'] > 25:
        #     suggestions.append("🟡 <b>PREVENTION:</b> Weight management to maintain healthy BMI")
        
        # if input_data['physical_activity'] == 0:
        #     suggestions.append("🟢 <b>LIFESTYLE:</b> Countinue regular physical activity")
    
    return suggestions

# --- Enhanced CSS ---
def load_css():
    """Load enhanced CSS styling."""
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px 20px;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        .section-header {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 15px 20px;
            border-radius: 10px;
            color: white;
            margin-top: 25px;
            margin-bottom: 15px;
            font-weight: 600;
        }
        .result-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin-top: 20px;
            border-left: 5px solid #4CAF50;
        }
        .high-risk {
            border-left-color: #f44336 !important;
            background: linear-gradient(135deg, #ffebee 0%, #fce4ec 100%);
        }
        .low-risk {
            border-left-color: #4caf50 !important;
            background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
        }
        .alert-card {
            background: linear-gradient(135deg, #ffcdd2 0%, #ef9a9a 100%);
            border-left: 5px solid #d32f2f;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
        }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            padding: 12px 24px !important;
            border-radius: 8px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            width: 100% !important;
            transition: all 0.3s ease !important;
        }
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin: 10px 0;
            text-align: center;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            color: #666;
            font-size: 0.9rem;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            border-top: 1px solid #eee;
            margin-top: 40px;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Main Application ---
def main():
    """Main application function."""
    global FEATURE_NAMES
    
    load_css()
    
    # Load model and encoders
    model, encoders, feature_names = load_model_and_encoders()
    FEATURE_NAMES = feature_names
    explainer = load_shap_explainer(_model=model)
    
    # Main header
    st.markdown('''
    <div class="main-header">
        <h3>🩺 AI-powered Hypertension Risk Prediction and Personalized Treatment Plan</h3>
        <p style="margin: 10px 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Disclaimer: This tool is for clinical decision support only and should complement, not replace, professional medical judgment.
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Information section
    with st.expander("ℹ️ About This Tool", expanded=False):
        st.markdown("""
        This application uses machine learning to assess hypertension risk based on patient clinical data. 
        
        **Features:**
        - 🤖 AI-powered risk prediction
        - 📊 SHAP explanations for model transparency
        - 🩺 Clinical guidelines-based recommendations
        - 📄 Comprehensive PDF reports
        - ⚠️ Real-time clinical alerts
        
        
        """)
    
    # Input Form
    st.markdown('<div class="section-header"><h2>👤 Patient Information</h2></div>', unsafe_allow_html=True)
    
    # Create input columns
    col1, col2 = st.columns(2)
    input_data = {}
    
    with col1:
        #st.subheader("📊 Vital Signs & Lab Values")
        input_data['age'] = st.number_input(
            "Age (years)", 
            min_value=FEATURE_CONFIG['age']['min'], 
            max_value=FEATURE_CONFIG['age']['max'], 
            value=FEATURE_CONFIG['age']['default'], 
            step=FEATURE_CONFIG['age']['step'],
            help="Patient's current age"
        )
        
        input_data['systolic_bp'] = st.number_input(
            "Systolic Blood Pressure (mmHg)", 
            min_value=FEATURE_CONFIG['systolic_bp']['min'], 
            max_value=FEATURE_CONFIG['systolic_bp']['max'], 
            value=FEATURE_CONFIG['systolic_bp']['default'], 
            step=FEATURE_CONFIG['systolic_bp']['step'],
            help="Normal: <120, Elevated: 120-129, Stage 1 HTN: 130-139, Stage 2 HTN: ≥140"
        )
        
        input_data['diastolic_bp'] = st.number_input(
            "Diastolic Blood Pressure (mmHg)", 
            min_value=FEATURE_CONFIG['diastolic_bp']['min'], 
            max_value=FEATURE_CONFIG['diastolic_bp']['max'], 
            value=FEATURE_CONFIG['diastolic_bp']['default'], 
            step=FEATURE_CONFIG['diastolic_bp']['step'],
            help="Normal: <80, Stage 1 HTN: 80-89, Stage 2 HTN: ≥90"
        )
        
        input_data['bmi'] = st.number_input(
            "Body Mass Index (BMI)", 
            min_value=FEATURE_CONFIG['bmi']['min'], 
            max_value=FEATURE_CONFIG['bmi']['max'], 
            value=FEATURE_CONFIG['bmi']['default'], 
            step=FEATURE_CONFIG['bmi']['step'],
            help="Normal: 18.5-24.9, Overweight: 25-29.9, Obese: ≥30"
        )
        
        input_data['glucose_level'] = st.number_input(
            "Glucose Level (mg/dL)", 
            min_value=FEATURE_CONFIG['glucose_level']['min'], 
            max_value=FEATURE_CONFIG['glucose_level']['max'], 
            value=FEATURE_CONFIG['glucose_level']['default'], 
            step=FEATURE_CONFIG['glucose_level']['step'],
            help="Normal: <100, Prediabetes: 100-125, Diabetes: ≥126"
        )
        
        input_data['cholesterol'] = st.number_input(
            "Total Cholesterol (mg/dL)", 
            min_value=FEATURE_CONFIG['cholesterol']['min'], 
            max_value=FEATURE_CONFIG['cholesterol']['max'], 
            value=FEATURE_CONFIG['cholesterol']['default'], 
            step=FEATURE_CONFIG['cholesterol']['step'],
            help="Desirable: <200, Borderline: 200-239, High: ≥240"
        )
    
    with col2:
        #st.subheader("🏥 Demographics & Risk Factors")
        
        # Gender selection
        if 'gender' in encoders and hasattr(encoders['gender'], 'classes_'):
            gender_options = encoders['gender'].classes_.tolist()
        else:
            gender_options = FEATURE_CONFIG['gender']['options']
        input_data['gender'] = st.selectbox("Gender", options=gender_options)
        
        # Binary risk factors with better layout
        input_data['smoker'] = st.radio(
            "Current Smoker", 
            options=FEATURE_CONFIG['smoker']['options'], 
            help="Current tobacco use status"
        )
        
        input_data['alcohol_use'] = st.radio(
            "Regular Alcohol Use", 
            options=FEATURE_CONFIG['alcohol_use']['options'],
            help="Regular alcohol consumption (≥3 drinks/week)"
        )
        
        input_data['physical_activity'] = st.radio(
            "Regular Physical Activity", 
            options=FEATURE_CONFIG['physical_activity']['options'],
            help="≥150 minutes moderate exercise per week"
        )
        
        input_data['med_adherence'] = st.radio(
            "Medication Adherence", 
            options=FEATURE_CONFIG['med_adherence']['options'],
            help="Takes medications as prescribed"
        )
    
    # Validate and process input data
    input_data = validate_input_data(input_data)
    
    # Generate clinical alerts
    alerts = get_clinical_alerts(input_data)
    
    # Display alerts if any
    if alerts:
        st.markdown('<div class="section-header"><h2>🚨 Clinical Alerts</h2></div>', unsafe_allow_html=True)
        for alert in alerts:
            st.markdown(f'<div class="alert-card">{alert}</div>', unsafe_allow_html=True)
    
    # Prediction section
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Risk", key="predict")
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Process prediction
    if predict_button:
        with st.spinner("🔄 Analyzing patient data..."):
            try:
                # Prepare data for prediction - use the correct feature order
                input_df = pd.DataFrame([input_data])
                
                # Ensure columns are in the correct order that matches the model
                input_df = input_df[FEATURE_NAMES]
                
                # Apply encoders
                input_df_processed = input_df.copy()
                for col, encoder in encoders.items():
                    if col in input_df_processed.columns:
                        try:
                            input_df_processed[col] = encoder.transform(input_df_processed[col])
                        except ValueError as e:
                            st.error(f"❌ Error encoding '{col}': {e}")
                            st.stop()
                
                # Make prediction
                prediction = model.predict(input_df_processed)[0]
                prediction_proba = model.predict_proba(input_df_processed)[0]
                
                # Generate SHAP explanation
                shap_fig = None
                if explainer:
                    try:
                        shap_values = explainer.shap_values(input_df_processed)
                        shap_values_for_class_1 = shap_values[1] if isinstance(shap_values, list) else shap_values
                        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                        
                        fig, ax = plt.subplots(figsize=(12, 8))
                        shap.waterfall_plot(
                            shap.Explanation(
                                values=shap_values_for_class_1[0],
                                base_values=base_value,
                                data=input_df_processed.iloc[0],
                                feature_names=FEATURE_NAMES
                            ),
                            show=False,
                            max_display=len(FEATURE_NAMES)
                        )
                        plt.tight_layout()
                        shap_fig = fig
                    except Exception as e:
                        logger.error(f"SHAP error: {e}")
                        st.warning(f"⚠️ Could not generate SHAP explanation: {e}")
                
                # Store results
                st.session_state.results = {
                    'input_data': input_data,
                    'prediction': prediction,
                    'prediction_proba': prediction_proba,
                    'shap_fig': shap_fig,
                    'alerts': alerts,
                    'feature_names': FEATURE_NAMES
                }
                
            except KeyError as e:
                st.error(f"❌ Feature mismatch error: {e}")
                st.error("🔧 This usually means the model expects different features or in a different order.")
                st.info("💡 Check that your model file matches the expected features.")
                logger.error(f"Feature mismatch: {e}")
                st.stop()
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
                logger.error(f"Prediction error: {e}")
                st.stop()
    
    # Display results
    if st.session_state.results:
        results = st.session_state.results
        input_data = results['input_data']
        prediction = results['prediction']
        prediction_proba = results['prediction_proba']
        shap_fig = results.get('shap_fig')
        alerts = results.get('alerts', [])
        feature_names = results.get('feature_names', FEATURE_NAMES)
        
        # Risk Assessment Results
        st.markdown('<div class="section-header"><h2>📊 Risk Assessment Results</h2></div>', unsafe_allow_html=True)
        
        # Main prediction display
        risk_text = "High Risk" if prediction == 1 else "Low Risk"
        risk_class = "high-risk" if prediction == 1 else "low-risk"
        risk_emoji = "🔴" if prediction == 1 else "🟢"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value" style="color: {'#f44336' if prediction == 1 else '#4caf50'}">
                    {risk_emoji} {risk_text}
                </div>
                <div class="metric-label">Predicted Risk Level</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{prediction_proba[0]:.1%}</div>
                <div class="metric-label">Probability of Low Risk</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{prediction_proba[1]:.1%}</div>
                <div class="metric-label">Probability of High Risk</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Detailed prediction card
        confidence_level = max(prediction_proba)
        confidence_text = "High" if confidence_level >= 0.8 else "Medium" if confidence_level >= 0.6 else "Low"
        
        st.markdown(f'''
        <div class="result-card {risk_class}">
            <h3>🎯 Detailed Risk Assessment</h3>
            <p><strong>Risk Classification:</strong> <span style="color: {'#d32f2f' if prediction == 1 else '#388e3c'}; font-weight: bold;">{risk_text}</span></p>
            <p><strong>Model Confidence:</strong> {confidence_text} ({confidence_level:.1%})</p>
            <p><strong>Risk Interpretation:</strong> 
            {
                "This patient shows elevated risk factors for hypertension and should receive immediate clinical attention and intervention." 
                if prediction == 1 
                else "This patient demonstrates favorable cardiovascular risk profile with low hypertension risk."
            }
            </p>
        </div>
        ''', unsafe_allow_html=True)
        
        # SHAP Explanation
        if shap_fig:
            st.markdown('<div class="section-header"><h2>🔍 AI Model Explanation (SHAP)</h2></div>', unsafe_allow_html=True)
            
            with st.expander("🤔 How to interpret this chart", expanded=False):
                st.markdown("""
                **SHAP (SHapley Additive exPlanations) Waterfall Plot:**
                
                - **Red bars (→):** Factors that *increase* hypertension risk
                - **Blue bars (←):** Factors that *decrease* hypertension risk  
                - **Bar length:** Shows the strength of each factor's influence
                - **E[f(X)]:** The average prediction across all patients (baseline)
                - **f(x):** The final prediction for this specific patient
                
                The chart reads from bottom to top, showing how each factor pushes the prediction toward or away from high risk.
                """)
            
            st.pyplot(shap_fig)
            plt.close(shap_fig)
        else:
            st.warning("⚠️ SHAP explanation not available for this prediction.")
        
        # Clinical Recommendations
        suggestions = get_treatment_suggestions(input_data, prediction, feature_names)
        
        st.markdown('<div class="section-header"><h2>💊 Treatment suggestions</h2></div>', unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown('''
            <div class="result-card high-risk">
                <h4>🚨 High-Risk Patient Management Plan</h4>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="result-card low-risk">
                <h4>✅ Low-Risk Patient Maintenance Plan</h4>
            </div>
            ''', unsafe_allow_html=True)
        
        # Display suggestions in organized categories
        priority_suggestions = [s for s in suggestions if "URGENT" in s or "HIGH PRIORITY" in s]
        lifestyle_suggestions = [s for s in suggestions if "LIFESTYLE" in s]
        medication_suggestions = [s for s in suggestions if "MEDICATION" in s]
        monitoring_suggestions = [s for s in suggestions if "MONITORING" in s or "REFERRAL" in s]
        other_suggestions = [s for s in suggestions if s not in priority_suggestions + lifestyle_suggestions + medication_suggestions + monitoring_suggestions]
        
        if priority_suggestions:
            st.markdown("**🔴 Immediate Actions:**")
            for suggestion in priority_suggestions:
                st.markdown(suggestion, unsafe_allow_html=True)
        
        if medication_suggestions:
            st.markdown("**💊 Pharmacological Interventions:**")
            for suggestion in medication_suggestions:
                st.markdown(suggestion, unsafe_allow_html=True)
        
        if lifestyle_suggestions:
            st.markdown("**🏃‍♂️ Lifestyle Modifications:**")
            for suggestion in lifestyle_suggestions:
                st.markdown(suggestion, unsafe_allow_html=True)
        
        if monitoring_suggestions:
            st.markdown("**📋 Monitoring & Follow-up:**")
            for suggestion in monitoring_suggestions:
                st.markdown(suggestion, unsafe_allow_html=True)
        
        if other_suggestions:
            for suggestion in other_suggestions:
                st.markdown(suggestion, unsafe_allow_html=True)
        
        # Risk Factor Analysis
        st.markdown('<div class="section-header"><h2>📈 Risk Factor Analysis</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 Current Risk Factors:**")
            risk_factors = []
            if input_data['systolic_bp'] >= 140 or input_data['diastolic_bp'] >= 90:
                risk_factors.append(f"• Hypertension (BP: {input_data['systolic_bp']}/{input_data['diastolic_bp']} mmHg)")
            if input_data['bmi'] >= 30:
                risk_factors.append(f"• Obesity (BMI: {input_data['bmi']:.1f})")
            if input_data['cholesterol'] > 200:
                risk_factors.append(f"• High cholesterol ({input_data['cholesterol']} mg/dL)")
            if input_data['glucose_level'] > 126:
                risk_factors.append(f"• Diabetes/Prediabetes ({input_data['glucose_level']} mg/dL)")
            if input_data['smoker'] == 1:
                risk_factors.append("• Current smoking")
            if input_data['physical_activity'] == 0:
                risk_factors.append("• Physical inactivity")
            if input_data['med_adherence'] == 0:
                risk_factors.append("• Poor medication adherence")
            
            if risk_factors:
                for rf in risk_factors:
                    st.markdown(rf)
            else:
                st.markdown("• No major risk factors identified")
        
        with col2:
            st.markdown("**🟢 Protective Factors:**")
            protective_factors = []
            if input_data['systolic_bp'] < 120 and input_data['diastolic_bp'] < 80:
                protective_factors.append("• Normal blood pressure")
            if input_data['bmi'] >= 18.5 and input_data['bmi'] < 25:
                protective_factors.append("• Healthy weight")
            if input_data['cholesterol'] <= 200:
                protective_factors.append("• Normal cholesterol")
            if input_data['glucose_level'] <= 100:
                protective_factors.append("• Normal glucose")
            if input_data['smoker'] == 0:
                protective_factors.append("• Non-smoker")
            if input_data['physical_activity'] == 1:
                protective_factors.append("• Regular physical activity")
            if input_data['med_adherence'] == 1:
                protective_factors.append("• Good medication adherence")
            
            if protective_factors:
                for pf in protective_factors:
                    st.markdown(pf)
            else:
                st.markdown("• Limited protective factors")
        
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📥 Generate PDF Report", key="export_pdf"):
                with st.spinner("🔄 Generating comprehensive PDF report..."):
                    try:
                        pdf_buffer = create_pdf_report(
                            input_data, 
                            prediction, 
                            prediction_proba, 
                            shap_fig,
                            suggestions,
                            alerts,
                            feature_names
                        )
                        
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"hypertension_risk_report_{timestamp}.pdf"
                        
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_buffer,
                            file_name=filename,
                            mime="application/pdf",
                            help="Click to download the complete risk assessment report"
                        )
                        
                        st.success("✅ PDF report generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating PDF: {e}")
                        logger.error(f"PDF generation error: {e}")

    # Additional Information
    with st.expander("📚 Clinical Guidelines & References", expanded=False):
        st.markdown("""
        **Hypertension Classification (AHA/ACC 2017):**
        - Normal: <120/80 mmHg
        - Elevated: 120-129/<80 mmHg  
        - Stage 1: 130-139/80-89 mmHg
        - Stage 2: ≥140/90 mmHg
        - Crisis: >180/120 mmHg
        
        **BMI Categories:**
        - Underweight: <18.5
        - Normal: 18.5-24.9
        - Overweight: 25-29.9
        - Obese: ≥30
        
        **Cholesterol Levels:**
        - Desirable: <200 mg/dL
        - Borderline: 200-239 mg/dL
        - High: ≥240 mg/dL
        
        **Glucose Levels:**
        - Normal: <100 mg/dL
        - Prediabetes: 100-125 mg/dL
        - Diabetes: ≥126 mg/dL
        """)
    
    
        

    # Footer
    st.markdown('''
    <div class="footer">
        <p>Powered by AI • Built with Streamlit • Designed for Healthcare Professionals</p>
        <p><small>Version 1.0 | Last Updated: 2025</small></p>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
                    