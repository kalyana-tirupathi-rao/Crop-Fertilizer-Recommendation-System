# -*- coding: utf-8 -*-
"""Train Crop and Fertilizer Models"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# --- Create 'models/' folder if not exists ---
if not os.path.exists("models"):
    os.makedirs("models")
    print("✅ 'models/' folder created successfully!")

# --- Load Datasets ---
crop_data = pd.read_csv('./datasets/Crop_recommendation.csv')
fertilizer_data = pd.read_csv('./datasets/Fertilizer_Prediction.csv')

# --- Train Crop Model ---
X_crop = crop_data.drop(columns=['label'])
y_crop = crop_data['label']

le_crop = LabelEncoder()
y_crop_encoded = le_crop.fit_transform(y_crop)

X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(
    X_crop, y_crop_encoded, test_size=0.2, random_state=42
)

crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(X_train_crop, y_train_crop)

# Save Crop Model
pickle.dump(crop_model, open("models/crop_model.pkl", "wb"))
pickle.dump(le_crop, open("models/le_crop.pkl", "wb"))
print("✅ Crop Model Saved!")

# --- Train Fertilizer Model ---
fertilizer_data.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

X_fert = fertilizer_data.drop(columns=['Fertilizer Name'])
y_fert = fertilizer_data['Fertilizer Name']

categorical_columns = ['Soil Type', 'Crop Type']
for col in categorical_columns:
    le = LabelEncoder()
    X_fert[col] = le.fit_transform(X_fert[col])

le_fert = LabelEncoder()
y_fert_encoded = le_fert.fit_transform(y_fert)

X_train_fert, X_test_fert, y_train_fert, y_test_fert = train_test_split(
    X_fert, y_fert_encoded, test_size=0.2, random_state=42
)

fert_model = RandomForestClassifier(n_estimators=100, random_state=42)
fert_model.fit(X_train_fert, y_train_fert)

# Save Fertilizer Model
pickle.dump(fert_model, open("models/fert_model.pkl", "wb"))
pickle.dump(le_fert, open("models/le_fert.pkl", "wb"))
print("✅ Fertilizer Model Saved!")

print("\n🎉 Models trained and saved successfully in 'models/' folder!")
