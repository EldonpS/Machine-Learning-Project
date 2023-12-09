import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path
OUTPUT_PATH = Path(__file__).parent
DATASET_PATH = OUTPUT_PATH / "dataset" / "Datas.csv"

df = pd.read_csv(DATASET_PATH)

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Try different values of k
k_values = range(1, 21)  # You can adjust the range based on your preference
accuracy_values = []

for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)

# Plot the accuracy values for different k
plt.plot(k_values, accuracy_values, marker='o')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('Accuracy for Different Values of k')
plt.show()

# Find the k with the highest accuracy
best_k = k_values[accuracy_values.index(max(accuracy_values))]
print("Best k:", best_k)
