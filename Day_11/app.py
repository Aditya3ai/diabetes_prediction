from flask import Flask, request, render_template
import numpy as np
from keras.models import model_from_json

app = Flask(__name__)

# Load the model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_features = [float(x) for x in request.form.values()]
    input_array = np.array([input_features])
    
    # Predict using the loaded model
    prediction = model.predict(input_array)
    result = (prediction[0][0] > 0.5).astype(int)  # Convert probability to binary class

    return render_template('index.html', 
                           prediction_text=f'Diabetes Prediction: {"Positive" if result == 1 else "Negative"}')

if __name__ == "__main__":
    app.run(debug=True)
