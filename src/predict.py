from flask import Flask, request, jsonify
import mlflow
import pandas as pd

app = Flask(__name__)

# Load model from MLflow Model Registry
logged_model = 'models:/CreditRiskLogisticModel/Production'
model = mlflow.pyfunc.load_model(logged_model)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        input_data = request.get_json()

        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict probability of high risk (class 1)
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]

        return jsonify({
            'prediction': int(prediction[0]),
            'risk_probability': round(probability, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
