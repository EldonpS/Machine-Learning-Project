import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model('trained_model.h5')

# Load the unlabeled data from another CSV file
unlabeled_df = pd.read_csv("Ready.csv")

# Assume the features are in the same format as your original data
X_unlabeled = unlabeled_df.values.reshape(unlabeled_df.shape[0], unlabeled_df.shape[1], 1)

# Make predictions on the unlabeled data using the loaded model
predictions_prob = loaded_model.predict(X_unlabeled)
predictions = np.argmax(predictions_prob, axis=1)

# If LabelEncoder was used during training, create a new instance and fit it on the original labeled data
label_encoder = LabelEncoder()
# Replace 'your_original_labeled_data.csv' with the file path of your original labeled data
original_labeled_data = pd.read_csv("Data_Train.csv")
y_original = original_labeled_data.iloc[:, -1].values
label_encoder.fit(y_original)

# Convert numerical predictions back to original labels
predicted_labels = label_encoder.inverse_transform(predictions)

# Display or use the predicted labels as needed
print(predicted_labels)
