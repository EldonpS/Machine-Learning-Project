import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent
DATASET_PATH = OUTPUT_PATH / "dataset" / "Datas.csv"

df = pd.read_csv(DATASET_PATH)

x = df.iloc[:, :-1].values
y_str = df.iloc[:, -1].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_str)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')  
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

start = time.time()
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
end = time.time()

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Time Computation : " + str(end - start))

cm = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_str), yticklabels=np.unique(y_str))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
