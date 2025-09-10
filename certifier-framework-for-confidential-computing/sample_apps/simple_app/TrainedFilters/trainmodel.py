import joblib

# Load the model or object
model = joblib.load("mlp_filter.joblib")

# Now you can use the model, e.g., for prediction
# Example for a scikit-learn model:
prediction = model.predict()
print(prediction)
