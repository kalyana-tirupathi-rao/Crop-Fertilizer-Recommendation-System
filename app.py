from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# --- Load Models ---
model_path = "models/"
if os.path.exists(model_path):
    try:
        with open(os.path.join(model_path, "crop_model.pkl"), 'rb') as file:
            crop_model = pickle.load(file)
        with open(os.path.join(model_path, "le_crop.pkl"), 'rb') as file:
            le_crop = pickle.load(file)
        with open(os.path.join(model_path, "fert_model.pkl"), 'rb') as file:
            fert_model = pickle.load(file)
        with open(os.path.join(model_path, "le_fert.pkl"), 'rb') as file:
            le_fert = pickle.load(file)
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
else:
    print("❌ 'models/' folder not found! Run train_models.py first.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        features = np.array([[
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]])

        crop_index = crop_model.predict(features)[0]
        crop_name = le_crop.inverse_transform([crop_index])[0]
        return render_template('index.html', crop_result=crop_name)
    except Exception as e:
        return render_template('index.html', crop_result=f"Error: {e}")

@app.route('/predict_fert', methods=['POST'])
def predict_fert():
    try:
        features = np.array([[
            float(request.form['soil_type']),
            float(request.form['crop_type']),
            float(request.form['temperature_fert']),
            float(request.form['humidity_fert']),
            float(request.form['moisture']),
            float(request.form['nitrogen']),
            float(request.form['phosphorus']),
            float(request.form['potassium'])
        ]])

        fert_index = fert_model.predict(features)[0]
        fert_name = le_fert.inverse_transform([fert_index])[0]
        return render_template('index.html', fert_result=fert_name)
    except Exception as e:
        return render_template('index.html', fert_result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
