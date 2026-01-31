#pip install streamlit pandas numpy matplotlib
streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("🫀 Heart Disease Prediction App")
st.markdown("Logistic Regression from scratch (Andrew Ng style)")

# --------------------------------
# Sidebar Controls
# --------------------------------
st.sidebar.header("Model Settings")
lambda_ = st.sidebar.slider("L2 Regularization (λ)", 0.0, 1.0, 0.5, 0.05)
threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.3, 0.05)
iterations = st.sidebar.slider("Gradient Descent Iterations", 200, 2000, 800, 200)
alpha = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.05, 0.005)

# --------------------------------
# Upload CSV
# --------------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    # Load & clean
    df = pd.read_csv(uploaded_file)
    df = df.dropna()

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    # Mean–Max normalization
    X = (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # Split 60/20/20
    m = len(X)
    i1, i2 = int(0.6*m), int(0.8*m)
    X_train, X_test = X[:i1], X[i2:]
    y_train, y_test = y[:i1], y[i2:]

    # --------------------------------
    # Logistic Regression (L2)
    # --------------------------------
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def train_l2(X, y):
        theta = np.zeros((X.shape[1], 1))
        m = len(y)

        for _ in range(iterations):
            h = sigmoid(X @ theta)
            error = h - y

            reg = (lambda_ / m) * theta
            reg[0] = 0  # bias excluded

            grad = (1 / m) * (X.T @ error) + reg
            theta -= alpha * grad

        return theta

    theta = train_l2(X_train, y_train)

    # Predictions
    y_prob = sigmoid(X_test @ theta)
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion Matrix
    TP = np.sum((y_pred==1)&(y_test==1))
    TN = np.sum((y_pred==0)&(y_test==0))
    FP = np.sum((y_pred==1)&(y_test==0))
    FN = np.sum((y_pred==0)&(y_test==1))

    def sd(a,b): return a/b if b!=0 else 0

    accuracy = sd(TP+TN, TP+TN+FP+FN)
    precision = sd(TP, TP+FP)
    recall = sd(TP, TP+FN)
    f1 = sd(2*precision*recall, precision+recall)
    specificity = sd(TN, TN+FP)

    # --------------------------------
    # Display Metrics
    # --------------------------------
    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("Precision", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{f1:.3f}")
    col5.metric("Specificity", f"{specificity:.3f}")
    col6.metric("TP / FN", f"{TP} / {FN}")

    st.write("**Confusion Matrix**")
    st.write({"TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN)})

    # --------------------------------
    # ROC Curve
    # --------------------------------
    thresholds = np.linspace(0, 1, 100)
    TPR, FPR = [], []

    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        tp = np.sum((yp==1)&(y_test==1))
        tn = np.sum((yp==0)&(y_test==0))
        fp = np.sum((yp==1)&(y_test==0))
        fn = np.sum((yp==0)&(y_test==1))

        TPR.append(sd(tp, tp+fn))
        FPR.append(sd(fp, fp+tn))

    auc = np.trapz(TPR, FPR)

    fig, ax = plt.subplots()
    ax.plot(FPR, TPR, label=f"AUC = {abs(auc):.3f}")
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    st.pyplot(fig)

else:
    st.info("Upload a CSV file to get started.")
