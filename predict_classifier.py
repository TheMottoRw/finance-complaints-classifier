import pandas as pd
import numpy as np
import joblib
import time
import signal
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Linear SVC": LinearSVC(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),  # Full complexity
    "Multinomial NB": MultinomialNB(),
    "SGD Classifier": SGDClassifier(loss="hinge", random_state=42)  # Linear SVM-like classifier
}
batch_accuracies = {model_name: [] for model_name in models.keys()}  # Store batch accuracy

# Global variables for signal handling
tfidf_vectorizer = None
all_classes_list = None

def predict_complaint(complaint_text, model_name="SGD Classifier"):
    """
    Predict the category of a complaint using the specified model.

    This function loads the specified model, TF-IDF vectorizer, and unique classes,
    transforms the complaint text using the vectorizer, and predicts the category
    using the model. It automatically finds the most recent version of each file.

    Args:
        complaint_text (str): The text of the complaint to classify
        model_name (str): Name of the model to use for prediction.
                          Must be one of: "Logistic Regression", "Linear SVC",
                          "Random Forest", "Multinomial NB", "SGD Classifier"

    Returns:
        str: Predicted category or error message if the model couldn't be loaded
             or if there was an error during prediction

        Credit card or prepaid card
    """
    try:
        import os
        import glob

        # Find the most recent model file
        # model_prefix = f"models/{model_name.lower().replace(' ', '_')}_model"
        model_prefix = f"models/sgd_classifier_model"
        model_files = glob.glob(f"{model_prefix}*.pkl")
        if not model_files:
            return f"No model files found for {model_name}"

        # Sort by modification time (newest first)
        model_filename = max(model_files, key=os.path.getmtime)

        # Find the most recent vectorizer file
        vectorizer_files = glob.glob("models/tfidf_vectorizer.pkl")
        if not vectorizer_files:
            return "No vectorizer files found"

        # Sort by modification time (newest first)
        vectorizer_filename = max(vectorizer_files, key=os.path.getmtime)

        # Find the most recent classes file
        classes_files = glob.glob("models/unique_predictable_class.pkl")
        if not classes_files:
            return "No classes files found"

        # Sort by modification time (newest first)
        classes_filename = max(classes_files, key=os.path.getmtime)

        # Load model, vectorizer and classes
        model = joblib.load(model_filename)
        tfidf = joblib.load(vectorizer_filename)
        all_classes = joblib.load(classes_filename)

        # Transform text
        complaint_tfidf = tfidf.transform([complaint_text])

        # Predict
        predicted_label = model.predict(complaint_tfidf)[0]

        return all_classes[predicted_label]
    except Exception as e:
        return f"Error predicting with {model_name}: {str(e)}"

if __name__ == "__main__":
    # Example new complaints for prediction
    # train_model()
    print(all_classes_list)
    new_complaints = [
        """
       Hello, I noticed an extra charge when I withdrew cash from the ATM. Can you please explain what this fee is for?
        """,
        "I need help transferring money to another account. The transaction failed but the amount was deducted",
        "My credit card is about to expire. I want to know how to renew it before it stops working",
        "Can I increase my daily withdrawal limit? I’m having trouble accessing enough cash for my business needs",
        "I’ve been charged interest on my credit card even though I paid on time. Please look into this issue",
        "I want to transfer money to another account. The transaction failed but the amount was deducted"
    ]

    # Predict using each model
    print("\nPredictions for sample complaint:")
    for i,complaint in enumerate(new_complaints):
        prediction = predict_complaint(complaint)
        print(f"Prediction {i}: {prediction}")

    # for model_name in models.keys():
    #     # Check if the model was trained (has accuracy scores)
    #     if model_name in batch_accuracies and len(batch_accuracies[model_name]) > 0:
    #         prediction = predict_complaint(new_complaints[0], model_name)
    #         print(f"{model_name}: {prediction}")
    #     else:
    #         print(f"{model_name}: Model was not trained or had no accuracy scores")
