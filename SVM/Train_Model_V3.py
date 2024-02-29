# 訓練模型
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

dataset_path = "dataset_V3"
label_file = "labels_V3.txt"

# Fetch all .npy files from the dataset directory
files = [f for f in os.listdir(dataset_path) if f.endswith('.npy')]

X_positions = []  # 用于存储手的位置坐标
X_distances = []  # 用于存储手指之间的距离
y = []

labels = []

# Load data and labels
for idx, file in enumerate(files):
    data = np.load(os.path.join(dataset_path, file))
    for entry in data:
        X_positions.append(entry[0])  # 手的位置坐标
        X_distances.append(entry[1])  # 手指之间的距离
        label = file.split('.')[0]
        labels.append(label)
        y.extend([idx] * len(data))

X_positions = np.array(X_positions)
X_distances = np.array(X_distances)
y = np.array(y)

# Save labels to labels.txt
with open(label_file, 'w') as f:
    for label in labels:
        f.write(label + "\n")

# Preprocess data
scaler_positions = StandardScaler()
X_positions_scaled = scaler_positions.fit_transform(X_positions)

scaler_distances = StandardScaler()
X_distances_scaled = scaler_distances.fit_transform(X_distances)

# Combine position coordinates and distances into a single feature matrix
X_combined = np.concatenate((X_positions_scaled, X_distances_scaled), axis=1)

# Train SVM model
clf = SVC(kernel='linear', probability=True)
clf.fit(X_combined, y)

# Save the model and the scalers
model_filename = "svm_model_V3.pkl"
joblib.dump(clf, model_filename)

scaler_positions_filename = "scaler_positions_V3.pkl"
joblib.dump(scaler_positions, scaler_positions_filename)

scaler_distances_filename = "scaler_distances_V3.pkl"
joblib.dump(scaler_distances, scaler_distances_filename)

print(f"Model saved to {model_filename}")
print(f"Position scaler saved to {scaler_positions_filename}")
print(f"Distance scaler saved to {scaler_distances_filename}")
print(f"Labels saved to {label_file}")
