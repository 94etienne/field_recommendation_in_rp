from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import pandas as pd
import os

# Load the saved model and encoders
model = load("./model/decision_tree_model.pkl")
encoder = load("./model/one_hot_encoder.pkl")
field_encoder = load("./model/label_encoder.pkl")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the path for the Excel file
EXCEL_FILE_PATH = "predictions_log.xlsx"

def preprocess_user_input(user_data, encoder):
    """
    Preprocess user input to match the model's input format.
    """
    # Convert user input into a DataFrame
    user_df = pd.DataFrame([user_data])

    # One-hot encode High_School_Stream
    stream_encoded = encoder.transform(user_df[["High_School_Stream"]])
    stream_encoded_df = pd.DataFrame(stream_encoded, columns=encoder.get_feature_names_out(["High_School_Stream"]))

    # Combine encoded features with subject scores
    user_processed = pd.concat([stream_encoded_df, user_df.iloc[:, 1:]], axis=1)

    return user_processed, user_df

def save_to_excel(user_data, predicted_field):
    """
    Save the input data and predicted field to an Excel file.
    """
    # Create a DataFrame with the input data and prediction
    data = {
        "High_School_Stream": [user_data["High_School_Stream"]],
        "Physics": [user_data["Physics"]],
        "Math": [user_data["Math"]],
        "Chemistry": [user_data["Chemistry"]],
        "Biology": [user_data["Biology"]],
        "Economics": [user_data["Economics"]],
        "Geography": [user_data["Geography"]],
        "History": [user_data["History"]],
        "Literature": [user_data["Literature"]],
        "Predicted_Field": [predicted_field]
    }
    df = pd.DataFrame(data)

    # Check if the Excel file already exists
    if os.path.exists(EXCEL_FILE_PATH):
        # Load the existing Excel file
        existing_df = pd.read_excel(EXCEL_FILE_PATH)
        # Append the new data
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        # Create a new Excel file
        updated_df = df

    # Save the updated DataFrame to Excel
    updated_df.to_excel(EXCEL_FILE_PATH, index=False)

@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint to make predictions.
    """
    # Get JSON data from the request
    user_data = request.json

    # Preprocess user input
    user_processed, user_df = preprocess_user_input(user_data, encoder)

    # Make prediction
    y_pred_encoded = model.predict(user_processed)
    y_pred = field_encoder.inverse_transform(y_pred_encoded)

    # Save the input and output to Excel
    save_to_excel(user_data, y_pred[0])

    # Return the prediction as JSON
    return jsonify({"predicted_field": y_pred[0]})

@app.route("/get_predictions", methods=["GET"])
def get_predictions():
    """
    API endpoint to fetch paginated predictions from the Excel file.
    """
    if os.path.exists(EXCEL_FILE_PATH):
        # Get pagination parameters from the request
        page = int(request.args.get("page", 1))  # Default to page 1
        per_page = int(request.args.get("per_page", 5))  # Default to 5 records per page
        predicted_field = request.args.get("predicted_field", None)  # Get the predicted field filter

        # Load the Excel file
        df = pd.read_excel(EXCEL_FILE_PATH)

        # Apply the predicted field filter if provided
        if predicted_field:
            df = df[df["Predicted_Field"] == predicted_field]

        # Calculate total number of records and pages
        total_records = len(df)
        total_pages = (total_records + per_page - 1) // per_page

        # Paginate the data
        start = (page - 1) * per_page
        end = start + per_page
        paginated_data = df.iloc[start:end].to_dict(orient="records")

        # Return paginated data and metadata
        return jsonify({
            "data": paginated_data,
            "total_records": total_records,
            "total_pages": total_pages,
            "current_page": page,
            "per_page": per_page
        })
    else:
        return jsonify({"data": [], "total_records": 0, "total_pages": 0, "current_page": 1, "per_page": 5})
if __name__ == "__main__":
    app.run(debug=True)