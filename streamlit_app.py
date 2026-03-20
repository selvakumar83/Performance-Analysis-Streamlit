import streamlit as st
import pandas as pd

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

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="ML Analysis", layout="wide")

# -------------------------
# SIMPLE CSS
# -------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #141E30, #243B55);
}
.title {
    text-align:center;
    color:#00FFD1;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOGIN DATA
# -------------------------
valid_students = ["23K81A7201","23K81A7202","23K81A7203","23K81A7204","23K81A7205",
    "23K81A7206","23K81A7207","23K81A7208","23K81A7210","23K81A7211",
    "23K81A7212","23K81A7213","23K81A7214","23K81A7215","23K81A7216",
    "23K81A7218","23K81A7219","23K81A7220","23K81A7222","23K81A7223",
    "23K81A7224","23K81A7225","23K81A7226","23K81A7227","23K81A7228",
    "23K81A7229","23K81A7230","23K81A7231","23K81A7232","23K81A7233",
    "23K81A7234","23K81A7235","23K81A7236","23K81A7237","23K81A7238",
    "23K81A7239","23K81A7240","23K81A7241","23K81A7242","23K81A7243",
    "23K81A7244","23K81A7245","23K81A7246","23K81A7247","23K81A7249",
    "23K81A7250","23K81A7251","23K81A7252","23K81A7253","23K81A7254",
    "23K81A7255","23K81A7256","23K81A7257","23K81A7258","23K81A7259",
    "23K81A7260","23K81A7261","23K81A7262","23K81A7263","23K81A7264",
    "24K85A7201","24K85A7202","24K85A7203","24K85A7204","24K85A7205",
    "24K85A7206"]

# -------------------------
# SESSION
# -------------------------
if "login" not in st.session_state:
    st.session_state.login = False

# -------------------------
# LOGIN PAGE
# -------------------------
if not st.session_state.login:

    st.markdown("<h1 class='title'>🔐 Student Login[III-A]</h1>", unsafe_allow_html=True)

    reg_no = st.text_input("Enter Your Roll Number")

    if st.button("Login"):
        if reg_no in valid_students:
            st.session_state.login = True
            st.session_state.user = reg_no
            st.success("Login Successful ✅")
            st.rerun()
        else:
            st.error("Invalid Roll Number ❌")

# -------------------------
# MAIN APP
# -------------------------
else:

    st.sidebar.success(f"Logged in: {st.session_state.user}")

    if st.sidebar.button("Logout"):
        st.session_state.login = False
        st.rerun()

    st.title("📊 Performance Analysis of Classification Algorithms")

    # -------------------------
    # AUTO LOAD CSV FROM GITHUB
    # -------------------------
    st.subheader("📂 Dataset Loaded Automatically")

    url = "https://raw.githubusercontent.com/selvakumar83/Performance-Analysis-Streamlit/main/sample_dataset.csv"

    try:
        data = pd.read_csv(url)
        data = data.dropna()

        st.success("Dataset Loaded from GitHub ✅")
        st.dataframe(data)

        # -------------------------
        # ML PROCESS
        # -------------------------
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
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

        # -------------------------
        # OUTPUT
        # -------------------------
        st.subheader("📊 Results")
        st.dataframe(df)

        st.subheader("📈 Accuracy Graph")
        st.bar_chart(df.set_index("Algorithm")["Accuracy"])

        best = df.loc[df['Accuracy'].idxmax()]
        st.success(f"🏆 Best Model: {best['Algorithm']} (Accuracy: {best['Accuracy']:.2f})")

    except:
        st.error("❌ Unable to load dataset. Check GitHub URL")
