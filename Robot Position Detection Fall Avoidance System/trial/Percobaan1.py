import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

# Load the dataset from CSV
df = pd.read_csv("Datas.csv")

# Assume the last column is the target variable and the rest are features
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode categorical labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Reshape input data for CNN (assuming it's tabular data)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# One-hot encode the target variable
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Initialize the CNN model
cnn_model = Sequential()

# Add Convolutional layers
cnn_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Save the trained model
cnn_model.save('trained_model.h5')

# Load the saved model
loaded_model = load_model('trained_model.h5')

# Assuming you have unlabeled data named X_unlabeled
# Reshape input data for CNN
X_unlabeled = X_unlabeled.reshape(X_unlabeled.shape[0], X_unlabeled.shape[1], 1)

# Make predictions on the unlabeled data using the loaded model
predictions_prob = loaded_model.predict(X_unlabeled)
predictions = np.argmax(predictions_prob, axis=1)

# Convert numerical predictions back to original labels if needed
predicted_labels = label_encoder.inverse_transform(predictions)

# Display or use the predicted labels as needed
print(predicted_labels)
