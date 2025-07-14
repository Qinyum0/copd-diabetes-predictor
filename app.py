#!/usr/bin/env python
# coding: utf-8

# In[4]:

import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path

# 页面配置
st.set_page_config(
    page_title="COPD-Diabetes Risk Prediction Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """安全加载模型函数"""
    try:
        model_path = Path(__file__).parent / "Diabetes_model.pkl"  # 建议文件名去掉空格
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# 只加载一次模型
model = load_model()

def predict_diabetes_risk(patient_data):
    """使用模型预测糖尿病风险"""
    try:
        # 转换分类变量为模型期望的格式
        processed_data = {
            'Age': patient_data['Age'],
            'Insurance_Medicare': 1 if patient_data['Insurance'] == 'Medicare' else 0,
            'Language_English': 1 if patient_data['Language'] == 'English' else 0,
            'Marital_Status_Married': 1 if patient_data['Marital_Status'] == 'Married' else 0,
            'Rdw': patient_data['Rdw'],
            'Total_Calcium': patient_data['Total_Calcium'],
            'Glucose': patient_data['Glucose'],
            'Glucocorticoids_Yes': 1 if patient_data['Glucocorticoids'] == 'Yes' else 0,
            'Nephrotoxic_Yes': 1 if patient_data['Nephrotoxic'] == 'Yes' else 0,
            'CKD_Yes': 1 if patient_data['CKD'] == 'Yes' else 0,
            'Hyperlipidemia_Yes': 1 if patient_data['Hyperlipidemia'] == 'Yes' else 0,
            'Heart_Failure_Yes': 1 if patient_data['Heart_Failure'] == 'Yes' else 0,
            'Pneumonia_Yes': 1 if patient_data['Pneumonia'] == 'Yes' else 0,
            'Drink_Wine_Yes': 1 if patient_data['Drink_Wine'] == 'Yes' else 0
        }
        
        input_df = pd.DataFrame([processed_data])
        proba = model.predict_proba(input_df)[0]
        prediction = model.predict(input_df)[0]
        return prediction, proba
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def main():
    st.title('COPD-Diabetes Risk Prediction Model')
    st.markdown("""
    This tool predicts the risk of diabetes in COPD patients.
    """)
    
    # 侧边栏输入
    st.sidebar.header('Patient Parameters')
    Age = st.sidebar.slider('Age', 18, 100, 50)
    Rdw = st.sidebar.number_input('RDW (%)', 0, 100, 14)
    Total_Calcium = st.sidebar.number_input('Total Calcium (mg/dL)', 0.0, 20.0, 9.0, step=0.1)
    Glucose = st.sidebar.number_input('Glucose (mg/dL)', 50, 1000, 100)
    
    Insurance = st.sidebar.selectbox('Insurance', ['Medicare', 'Other'])
    Language = st.sidebar.selectbox('Language', ['English', 'Other'])
    Marital_Status = st.sidebar.selectbox('Marital Status', ['Married', 'Other'])
    
    Glucocorticoids = st.sidebar.radio('Glucocorticoids Use', ['No', 'Yes'])
    Nephrotoxic = st.sidebar.radio('Nephrotoxic Drugs', ['No', 'Yes'])
    CKD = st.sidebar.radio('Chronic Kidney Disease', ['No', 'Yes'])
    Hyperlipidemia = st.sidebar.radio('Hyperlipidemia', ['No', 'Yes'])
    Heart_Failure = st.sidebar.radio('Heart Failure', ['No', 'Yes'])
    Pneumonia = st.sidebar.radio('Pneumonia History', ['No', 'Yes'])
    Drink_Wine = st.sidebar.radio('Alcohol Consumption', ['No', 'Yes'])
    
    if st.sidebar.button('Predict Diabetes Risk'):
        patient_data = {
            'Age': Age,
            'Insurance': Insurance,
            'Language': Language,
            'Marital_Status': Marital_Status,
            'Rdw': Rdw,
            'Total_Calcium': Total_Calcium,
            'Glucose': Glucose,
            'Glucocorticoids': Glucocorticoids,
            'Nephrotoxic': Nephrotoxic,
            'CKD': CKD,
            'Hyperlipidemia': Hyperlipidemia,
            'Heart_Failure': Heart_Failure,
            'Pneumonia': Pneumonia,
            'Drink_Wine': Drink_Wine
        }
        
        prediction, proba = predict_diabetes_risk(patient_data)
        
        if prediction is not None:
            st.subheader('Prediction Results')
            col1, col2 = st.columns([1, 3])
            
            with col1:
                risk_level = "High" if prediction == 1 else "Low"
                st.metric("Risk Level", value=risk_level)
                st.metric("Diabetes Probability", f"{proba[1]*100:.1f}%")
            
            with col2:
                st.progress(proba[1])
                if prediction == 1:
                    st.error('High risk of diabetes (probability ≥50%)')
                else:
                    st.success('Low risk of diabetes (probability <50%)')

if __name__ == '__main__':
    main()


# In[5]:


# get_ipython().system('jupyter nbconvert --to script "2型糖尿病模型网站1.0.ipynb"')


# In[ ]:




