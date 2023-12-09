import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent
DATASET_PATH = OUTPUT_PATH / "dataset" / "Datas.csv"

df = pd.read_csv(DATASET_PATH)

numerical_features = df.select_dtypes(include=['number']).values
text_columns = df.select_dtypes(include=['object']).columns

if len(text_columns) > 0:
    vectorizers = {}
    text_features_transformed = []

    for col in text_columns:
        vectorizer = CountVectorizer()
        text_features_col = df[col].values
        text_features_transformed_col = vectorizer.fit_transform(text_features_col).toarray()
        vectorizers[col] = vectorizer
        text_features_transformed.append(text_features_transformed_col)

    text_features_transformed = np.concatenate(text_features_transformed, axis=1)

    scaler = MinMaxScaler()
    numerical_features_scaled = scaler.fit_transform(numerical_features)

    x = np.concatenate((numerical_features_scaled, text_features_transformed), axis=1)
else:
    x = numerical_features

y = df.iloc[:, -1].values

start = time.time()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

nb_classifier = MultinomialNB()

f1_scorer = make_scorer(f1_score, average='weighted')
cv_scores = cross_val_score(nb_classifier, X_train, y_train, cv=5, scoring=f1_scorer)

nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)
end = time.time()

cv_f1_score = np.mean(cv_scores)
print("Cross-validated F1-score:", cv_f1_score)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("Time Computation:", end - start)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nb_classifier.classes_, yticklabels=nb_classifier.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
values = [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted'), recall_score(y_test, y_pred, average='weighted'), cv_f1_score]

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
plt.ylabel('Score')
plt.title('Model Evaluation Metrics')
plt.show()
