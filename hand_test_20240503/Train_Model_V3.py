# -*- coding: utf-8 -*-
# 訓練模型
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# ------------------------------------------------
# Change this to the topic you want to train

model_name = "topic_1"
question="question_8"
# ------------------------------------------------

# Check if the 'model' directory exists, if not, create it
if not os.path.exists( f"model/{model_name}/{question}"):
    os.makedirs( f"model/{model_name}/{question}")
    
dataset_path = f"dataset/{model_name}/{question}"
label_file = f"model/{model_name}/{question}/labels_topic.txt"

# Fetch all .npy files from the dataset directory
files = [f for f in os.listdir(dataset_path) if f.endswith('.npy')]

X = []
y = []

labels = []

# Load data and labels
for idx, file in enumerate(files):
    data = np.load(os.path.join(dataset_path, file))
    X.extend(data)  # Note: using extend instead of append
    label = file.split('.')[0]
    labels.append(label)
    y.extend([idx] * data.shape[0])

X = np.array(X)
y = np.array(y)

# Save labels to labels.txt
# with open(label_file, 'w') as f:
with open(label_file, 'w',encoding="UTF-8") as f:
     for label in labels:
        f.write(label + "\n")

# Preprocess data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train SVM model
clf = SVC(kernel='linear', probability=True)
clf.fit(X, y)

# Save the model and the scaler
model_filename = f"model/{model_name}/{question}/svm_model_topic.pkl"
joblib.dump(clf, model_filename)
scaler_filename = f"model/{model_name}/{question}/scaler_topic.pkl"
joblib.dump(scaler, scaler_filename)

print(f"Model saved to {model_filename}")
print(f"Scaler saved to {scaler_filename}")
print(f"Labels saved to {label_file}")