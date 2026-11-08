import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Model Evaluator", layout="wide")

st.title("🧠 End-to-End Classification Model Evaluator")
st.write("This application loads pre-trained models. If pre-trained models are missing, it dynamically reconstructs them using a fallback seed training process.")

# Sidebar options
st.sidebar.header("Configuration")

# Model Selection
model_option = st.sidebar.selectbox(
    'Choose Classifier Model',
    ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Support Vector Machine']
)

# Check if joblib model exists; if not, rebuild it dynamically!
model_filename = f"model/{model_option.lower().replace(' ', '_')}.joblib"
model = None

if os.path.exists(model_filename):
    model = joblib.load(model_filename)
    st.sidebar.success(f"Loaded {model_option} model from disk!")
else:
    st.sidebar.warning(f"{model_option} file not found. Re-training model dynamically on standard dataset...")
    
    # Rebuild baseline classification data matching the parameters
    X, y = make_classification(
        n_samples=1000,
        n_features=12,
        n_informative=8,
        n_redundant=4,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    fallback_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(probability=True, random_state=42)
    }
    
    model = fallback_models[model_option]
    model.fit(X_train, y_train)
    st.sidebar.success(f"Dynamically trained {model_option} successfully!")

# Dataset Upload
uploaded_file = st.file_uploader("Upload Test CSV dataset", type=["csv"])

if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Dataset Preview")
        st.dataframe(test_data.head())
        
        if 'target' not in test_data.columns:
            st.error("The uploaded CSV must contain a column named 'target'.")
        else:
            X_val = test_data.drop(columns=['target'])
            y_val = test_data['target']
            
            # Inference
            preds = model.predict(X_val)
            probas = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else preds
            
            # Metrics
            acc = accuracy_score(y_val, preds)
            auc = roc_auc_score(y_val, probas)
            prec = precision_score(y_val, preds)
            rec = recall_score(y_val, preds)
            f1 = f1_score(y_val, preds)
            mcc = matthews_corrcoef(y_val, preds)
            
            st.subheader(f"📊 Evaluation Metrics for {model_option}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc:.4f}")
            col1.metric("AUC Score", f"{auc:.4f}")
            col2.metric("Precision", f"{prec:.4f}")
            col2.metric("Recall", f"{rec:.4f}")
            col3.metric("F1 Score", f"{f1:.4f}")
            col3.metric("MCC Score", f"{mcc:.4f}")
            
            # Confusion Matrix
            st.subheader("📈 Confusion Matrix & Classification Report")
            cc1, cc2 = st.columns(2)
            
            with cc1:
                fig, ax = plt.subplots(figsize=(4, 3))
                cm = confusion_matrix(y_val, preds)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                ax.set_xlabel("Predicted Labels")
                ax.set_ylabel("True Labels")
                st.pyplot(fig)
                
            with cc2:
                report = classification_report(y_val, preds, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
                
    except Exception as e:
        st.error(f"An error occurred while parsing the dataset: {e}")
else:
    st.info("Please upload a test CSV dataset to begin evaluation.")
