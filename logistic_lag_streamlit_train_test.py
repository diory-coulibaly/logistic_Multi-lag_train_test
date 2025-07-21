import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.model_selection import train_test_split

st.title("📈 Predict GOOG Direction with Lagged Features allows training/test split")

st.sidebar.header("1. Upload Files")
goog_file = st.sidebar.file_uploader("GOOG CSV (semicolon-separated)", type="csv")
sp500_file = st.sidebar.file_uploader("S&P500 CSV (semicolon-separated)", type="csv")

st.sidebar.header("2. Model Options")
lags = st.sidebar.multiselect("Select lags (days)", [1, 2, 3, 5, 10], default=[1, 2, 5])
model_type = st.sidebar.selectbox("Model", ["Logistic Regression", "Decision Tree", "Random Forest"])
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=50, value=30) / 100

run_button = st.sidebar.button("Run Model")


def load_data(gfile, sfile):
    goog = pd.read_csv(gfile, sep=";")[["Date", "Adj.Close"]].rename(columns={"Adj.Close": "goog_price"})
    sp500 = pd.read_csv(sfile, sep=";")[["Date", "Adj.Close"]].rename(columns={"Adj.Close": "sp_price"})
    for df in [goog, sp500]:
        for col in df.columns:
            if col != 'Date':
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            
    df = pd.merge(goog, sp500, on="Date")
    df["Date"] = pd.to_datetime(df["Date"], format='%d/%m/%Y')
    df = df.sort_values("Date", ascending=False).reset_index(drop=True)
    
    df["goog_ret"] = df["goog_price"].pct_change()
    df["sp_ret"] = df["sp_price"].pct_change()
    return df.dropna().reset_index(drop=True)


def prepare_features(df, lags):
    df = df.copy()
    for lag in lags:
        df[f"goog_lag{lag}"] = df["goog_ret"].shift(-lag)
        df[f"sp_lag{lag}"] = df["sp_ret"].shift(-lag)
        df["goog_up"] = (df["goog_ret"] >= 0).astype(int)
        df = df.dropna().reset_index(drop=True)
    features = [f"goog_lag{lag}" for lag in lags] + [f"sp_lag{lag}" for lag in lags]
    return df, features


def select_model(name):
    if name == "Logistic Regression":
        return LogisticRegression()
    elif name == "Decision Tree":
        return DecisionTreeClassifier()
    elif name == "Random Forest":
        return RandomForestClassifier()


if run_button and goog_file and sp500_file:
    df_raw = load_data(goog_file, sp500_file)
    df_lagged, features = prepare_features(df_raw, lags)

    X = df_lagged[features]
    y = df_lagged["goog_up"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    model = select_model(model_type)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    st.subheader("📊 Performance Metrics")
    st.write(f"**Accuracy:** {acc:.3f}")
    st.write(f"**Precision:** {prec:.3f}")
    st.write(f"**Recall:** {rec:.3f}")
    st.write(f"**F1 Score:** {f1:.3f}")
    st.write(f"**AUC:** {auc:.3f}")

    st.subheader("📉 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(plt.gcf())

    st.subheader("📥 Download Prediction Results")
    df_out = df_lagged.iloc[y_test.index].copy()
    df_out["predicted"] = y_pred
    df_out["probability"] = y_proba
    csv = df_out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name="model_predictions.csv")

else:
    st.info("⬅️ Upload files, select options, and click 'Run Model'.")
