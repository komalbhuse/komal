import pickle
from flask import Flask, request, jsonify

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return "Welcome to the ML App!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.json

        # Make a prediction using the model
        prediction = model.predict([data["features"]])

        # Return the prediction as JSON
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Entry point for running the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



 
