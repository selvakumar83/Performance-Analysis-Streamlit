import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

st.set_page_config(page_title="ML Performance Analysis", layout="wide")

st.title("🚀 Performance Analysis of Classification Algorithms")

st.sidebar.header("Settings")
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20)

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:

    data = pd.read_csv(uploaded_file)
    data = data.dropna()

    st.subheader("Dataset Preview")
    st.dataframe(data)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(),
        'Random Forest': RandomForestClassifier(),
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='macro', zero_division=0)
        rec = recall_score(y_test, preds, average='macro', zero_division=0)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)

        results.append([name, acc, prec, rec, f1])

    df = pd.DataFrame(results, columns=["Algorithm","Accuracy","Precision","Recall","F1"])

    st.subheader("Results")
    st.dataframe(df)

    st.subheader("Accuracy Graph")
    st.bar_chart(df.set_index("Algorithm")["Accuracy"])

    best = df.loc[df['Accuracy'].idxmax()]
    st.success(f"Best Model: {best['Algorithm']} (Accuracy: {best['Accuracy']:.2f})")

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results", csv, "results.csv")

else:
    st.info("Upload a CSV file to start")