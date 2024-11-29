import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier  # Train and save a model

# Define the function to train and save the model
def train_model():
    try:
        # Load dataset
        data = load_iris()
        X, y = data.data, data.target

        # Train a simple model
        model = RandomForestClassifier()
        model.fit(X, y)

        # Save the model to a file
        with open("model.pkl", "wb") as file:
            pickle.dump(model, file)

        print("Model created and saved as 'model.pkl'")
    except Exception as e:
        print(f"Error: {e}")

# Entry point for the script
if __name__ == "__main__":
    train_model()
 
