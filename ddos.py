import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset (Replace with your actual file path)
file_path = "D:/ddos/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit(1)

# Display first few rows to understand the structure
print(df.head())

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Print column names to debug
print("Columns in the dataset:", df.columns)

# Drop irrelevant columns (if any) - Adjust based on dataset
columns_to_drop = ['Flow Bytes/s', 'Flow Packets/s']  # Example, modify as needed
df = df.drop(columns=columns_to_drop, errors='ignore')

# Handling missing values (if any)
df = df.dropna()

# Check if 'Label' column exists
if 'Label' not in df.columns:
    print("Error: 'Label' column not found in the dataset")
    exit(1)

# Encode Labels (Normal: 0, Attack: 1)
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

# Select numeric features for training
features = df.drop(columns=['Label'])
labels = df['Label']

# Normalize feature values
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Reshape features for ML/DL models
X = np.array(features_scaled)
y = np.array(labels)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Save trained model
joblib.dump(rf_model, "ddos_detection_model.pkl")
print("✅ Model Training Completed and Saved!")

# Confusion Matrix Heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.savefig('confusion_matrix_heatmap.png')
plt.close()

# Feature Importance Bar Plot
feature_importances = rf_model.feature_importances_
features_list = features.columns
sorted_idx = np.argsort(feature_importances)

plt.figure(figsize=(10, 7))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [features_list[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance Bar Plot')
plt.savefig('feature_importance_bar_plot.png')
plt.close()