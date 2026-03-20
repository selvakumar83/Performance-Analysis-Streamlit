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
valid_students = ["101", "102", "103", "104", "105"]

# -------------------------
# SESSION
# -------------------------
if "login" not in st.session_state:
    st.session_state.login = False

# -------------------------
# LOGIN PAGE
# -------------------------
if not st.session_state.login:

    st.markdown("<h1 class='title'>🔐 Student Login</h1>", unsafe_allow_html=True)

    reg_no = st.text_input("Enter Register Number")

    if st.button("Login"):
        if reg_no in valid_students:
            st.session_state.login = True
            st.session_state.user = reg_no
            st.success("Login Successful ✅")
            st.rerun()
        else:
            st.error("Invalid Register Number ❌")

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

    url = "https://github.com/selvakumar83/Performance-Analysis-Streamlit/edit/main/sample_dataset.csv"

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
