import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
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
from models.database import Result, Session

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['dataset']
        if file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)
            return redirect(url_for('analyze', filename=file.filename))
    return render_template('index.html')

@app.route('/analyze/<filename>')
def analyze(filename):
    # Read and clean the dataset
    data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    data = data.dropna()  # Remove any missing values

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Encode labels if necessary
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    algorithms = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(),
        'Random Forest': RandomForestClassifier(),
        'Naive Bayes': GaussianNB(),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier()
    }

    session = Session()
    results = []

    for name, model in algorithms.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='macro')
        rec = recall_score(y_test, preds, average='macro')
        f1 = f1_score(y_test, preds, average='macro')

        results.append((name, acc, prec, rec, f1))

        # Save only accuracy in database
        session.add(Result(algorithm=name, accuracy=acc))
    
    session.commit()
    session.close()

    # Save results to CSV file
    result_df = pd.DataFrame(results, columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    result_df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv'), index=False)

    return render_template('results.html', results=results)

if __name__ == '__main__':
   app.run(debug=True, use_reloader=False)

   

  
