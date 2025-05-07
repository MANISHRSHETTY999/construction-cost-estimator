from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and features
model = joblib.load('random_forest_construction_model.pkl')
features = joblib.load('construction_model_features.pkl')

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read input values
        inputs = []
        for feature in features:
            value = float(request.form.get(feature))
            inputs.append(value)
        
        input_array = np.array([inputs])
        prediction = model.predict(input_array)[0]
        
        return render_template('index.html', prediction_text=f'Estimated Project Cost: ₹{prediction:,.2f}', features=features)
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", features=features)

if __name__ == "__main__":
    app.run(debug=True)
