import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent
DATASET_PATH = OUTPUT_PATH / "dataset" / "Datas.csv"

df = pd.read_csv(DATASET_PATH)

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

start = time.time()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

knn_classifier = KNeighborsClassifier(n_neighbors=23)

knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)
end = time.time()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Time Computation:", end - start)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=knn_classifier.classes_, yticklabels=knn_classifier.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
