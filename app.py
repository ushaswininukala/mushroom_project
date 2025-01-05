from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

app = Flask(__name__)

# Paths for dataset and model
DATASET_PATH = r"C:\Users\ushas\Documents\mushroom_cleaned.csv"
MODEL_PATH = "mushroom_model.pkl"

# Load or train the model
if not os.path.exists(MODEL_PATH):
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Please ensure the file exists.")
    
    # Load the dataset
    df = pd.read_csv(DATASET_PATH)
    print("Dataset shape:", df.shape)

    # Train the model
    X = df.drop(columns=['class'])  # Ensure 'class' matches your target column
    y = df['class']
    model = RandomForestClassifier()
    model.fit(X, y)

    # Save the trained model
    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved at:", MODEL_PATH)
else:
    print("Model already exists. Loading from file...")

# Load the model
model = joblib.load(MODEL_PATH)

# Example route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction (POST request)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user inputs from form
        cap_diameter = float(request.form['cap-diameter'])
        cap_shape = int(request.form['cap-shape'])
        gill_attachment = int(request.form['gill-attachment'])
        gill_color = int(request.form['gill-color'])
        stem_height = float(request.form['stem-height'])
        stem_width = float(request.form['stem-width'])
        stem_color = int(request.form['stem-color'])
        season = float(request.form['season'])

        # Prepare the input features in the correct order
        input_features = [cap_diameter, cap_shape, gill_attachment, gill_color, stem_height, stem_width, stem_color, season]
        feature_columns = ['cap-diameter', 'cap-shape', 'gill-attachment', 'gill-color', 'stem-height', 'stem-width', 'stem-color', 'season']
        input_data = pd.DataFrame([input_features], columns=feature_columns)

        # Predict toxicity
        prediction = model.predict(input_data)
        result = "Toxic" if prediction[0] == 1 else "Non-Toxic"

        # Render the result on a new page
        return render_template('result.html', result=result)

    except KeyError as e:
        # Handle missing input fields
        return jsonify({"error": f"Missing input field: {e}"}), 400

    except ValueError as e:
        # Handle invalid input formats
        return jsonify({"error": f"Invalid input format: {e}"}), 400

if __name__ == '__main__':
    app.run(debug=True)
