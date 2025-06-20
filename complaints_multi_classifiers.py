"""
Complaints Classification with Multiple Models

This script trains multiple classification models on a complaints dataset using batch processing.
It implements the following classifiers:
- Logistic Regression
- Linear SVC
- Random Forest
- Multinomial Naive Bayes
- SGD Classifier (Linear SVM-like)

The script processes the dataset in batches to optimize resource usage, and saves all models,
the TF-IDF vectorizer, and the unique predictable classes for later prediction.

Usage:
1. Training: Run the script to train all models and save them to the 'models' directory.
   - You can customize which models to train by modifying the 'train_models' dictionary.
   - Models that take too long (> 30 seconds per batch) are automatically disabled.

2. Prediction: Use the 'predict_complaint' function to predict the category of a new complaint.
   - You can specify which model to use for prediction.
   - If a model wasn't trained or had no accuracy scores, an error message will be displayed.

"""

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

def save_models_with_timestamp(models_dict, tfidf, all_classes, batch_accuracies):
    """
    Save all models with a timestamp in the filename.

    Args:
        models_dict (dict): Dictionary of model name to model object
        tfidf: TF-IDF vectorizer
        all_classes: List of unique classes
        batch_accuracies (dict): Dictionary of model name to list of batch accuracies
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save models
    for model_name, model in models_dict.items():
        # Only save models that have been trained (have accuracy scores)
        if model_name in batch_accuracies and len(batch_accuracies[model_name]) > 0:
            model_filename = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, model_filename)
            print(f"Saved {model_name} as {model_filename}")

    # Save vectorizer and classes
    if tfidf is not None:
        joblib.dump(tfidf, f"models/tfidf_vectorizer.pkl")
        print(f"Saved TF-IDF vectorizer as models/tfidf_vectorizer.pkl")

    if all_classes is not None:
        joblib.dump(all_classes, f"models/unique_predictable_class.pkl")
        print(f"Saved unique classes as models/unique_predictable_class.pkl")

    print(f"All models and data saved with timestamp {timestamp}")

def signal_handler(sig, frame):
    """
    Handle Ctrl+C (SIGINT) by saving models with timestamp before exiting.
    """
    print("\nCtrl+C detected! Saving models before exiting...")
    save_models_with_timestamp(models, tfidf_vectorizer, all_classes_list, batch_accuracies)
    print("Models saved. Exiting gracefully.")
    exit(0)


def train_model():
    """
    Train multiple classifier models on complaint data using batch processing.

    This function:
    1. Processes the dataset in chunks to handle large datasets efficiently
    2. Trains multiple classifier models (if enabled in train_models dict)
    3. Evaluates model performance on each batch

    When does training stop:
    - When all data chunks are processed
    - Early stopping: If there's no improvement in average accuracy for a number of consecutive 
      batches (defined by 'patience' parameter, default is 3)
    - When Ctrl+C is pressed (models are saved before exiting)

    When are models saved:
    - Periodically: Every N batches (defined by 'save_interval' parameter, default is 5)
    - When a new best average accuracy is achieved
    - At the end of training (after all batches are processed)
    - When Ctrl+C is pressed (via signal handler)
    - When early stopping is triggered

    Returns:
        None
    """
    global models, batch_accuracies, tfidf_vectorizer, all_classes_list

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    # Load dataset in chunks
    dataset_path = "./datasets/complaints.csv"
    chunk_size = 512  # Adjust based on memory availability

    # Initialize TF-IDF Vectorizer (Fitted only once)
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
    tfidf_vectorizer = tfidf  # Update global variable

    # Initialize models for online learning


    # Flag to control which models to train (set to False to skip a model)
    train_models = {
        "Logistic Regression": False,
        "Linear SVC": False,
        "Random Forest": False,
        "Multinomial NB": False,
        "SGD Classifier": True
    }

    first_batch = True  # Track first batch
    all_classes = set()  # To collect all unique product classes

    # Dictionary for class mapping
    dict_y_to_replace = {
        'Credit reporting or other personal consumer reports': 'Credit reporting',
        'Credit reporting, credit repair services, or other personal consumer reports': 'Credit reporting',
        'Credit card': 'Credit card or prepaid card',
        'Prepaid card': 'Credit card or prepaid card',
        'Debt or credit management': 'Debt collection,credit management',
        'Debt collection': 'Debt collection,credit management',
        'Bank account': 'Bank account,checking and saving account or service',
        'Money transfer': 'Money transfer, virtual currency, or money service',
        'Student loan': 'Consumer loan',
        'Payday loan, title loan, or personal loan': 'Consumer loan',
        'Payday loan, title loan, personal loan, or advance loan': 'Consumer loan',
        'Mortgage': 'Consumer loan',
        'Vehicle loan': 'Consumer loan',
        'Vehicle loan or lease': 'Consumer loan',
        'Other financial service': 'Other financial service',
        'Virtual currency': 'Money transfer, virtual currency, or money service'
    }

    # Read dataset in chunks to get all unique classes
    all_classes = set()  # Initialize as set
    for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
        if "Product" in chunk and "Consumer complaint narrative" in chunk:
            all_classes.update(set(chunk["Product"].dropna().unique()))  # Update set with unique labels

    # Convert to sorted list for consistent encoding
    all_classes = sorted(list(all_classes))
    all_classes_df = pd.DataFrame({"Product": all_classes})
    all_classes_df["Product"] = all_classes_df["Product"].replace(dict_y_to_replace)
    all_classes = all_classes_df.Product.unique()
    all_classes_list = all_classes  # Update global variable
    print(f"Total unique classes found: {len(all_classes)}")
    joblib.dump(all_classes, "models/unique_predictable_class.pkl")

    # Read dataset in chunks again for training
    print("Training models in batches...")
    reader = pd.read_csv(dataset_path, chunksize=chunk_size)

    # Process all batches in the dataset
    batch_count = 0
    save_interval = 5  # Save models every 5 batches
    patience = 3  # Early stopping patience (number of batches without improvement)
    best_avg_acc = 0.0
    no_improvement_count = 0

    for chunk in reader:
        batch_count += 1
        print(f"Processing batch {batch_count}...")
        # Filter and clean data
        df1 = chunk[['Product', 'Consumer complaint narrative']].dropna()
        df1.columns = ['Product', 'Consumer_complaint']
        df1['Product'] = df1['Product'].replace(dict_y_to_replace)

        # If chunk is too small, skip it
        if df1.shape[0] < 2:  # Minimum 2 samples needed for splitting
            print("Skipping small chunk...")
            continue

        # Check if there are at least 2 unique classes in the chunk
        if len(df1["Product"].unique()) < 2:
            print("Skipping chunk with only one class...")
            continue

        # Encode labels using previously collected classes
        df1["Product"] = df1["Product"].astype("category").cat.set_categories(all_classes)
        y_encoded = df1["Product"].cat.codes  # Convert categories to numerical labels

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            df1["Consumer_complaint"], y_encoded, test_size=0.25, random_state=42
        )

        # TF-IDF Transformation
        if first_batch:
            X_train_tfidf = tfidf.fit_transform(X_train)  # Fit TF-IDF only once
            first_batch = False
        else:
            X_train_tfidf = tfidf.transform(X_train)  # Use transform after first batch

        X_test_tfidf = tfidf.transform(X_test)  # Always transform test data

        # Train and evaluate each model
        for model_name, model in models.items():
            # Skip models that are set to False in train_models
            if not train_models.get(model_name, True):
                print(f"Skipping {model_name} as specified in train_models")
                continue

            start_time = time.time()

            try:
                # Check if there are at least 2 unique classes in the training set
                if len(np.unique(y_train)) < 2:
                    print(f"Skipping {model_name} for this batch - not enough unique classes in training set")
                    continue

                # Train model
                if model_name == "SGD Classifier":
                    # SGD supports partial_fit for online learning
                    model.partial_fit(X_train_tfidf, y_train, classes=np.arange(len(all_classes)))
                else:
                    # Other models need to be fit from scratch each time
                    model.fit(X_train_tfidf, y_train)

                # Predict batch
                y_pred = model.predict(X_test_tfidf)

                # Calculate batch accuracy
                batch_acc = accuracy_score(y_test, y_pred)
                batch_accuracies[model_name].append(batch_acc)

                training_time = time.time() - start_time
                print(f"{model_name} - Batch Accuracy: {batch_acc:.4f}, Training Time: {training_time:.2f}s")

                # If a model takes too long (> 30 seconds), disable it for future batches
                if training_time > 30:
                    print(f"Model {model_name} took too long ({training_time:.2f}s). Disabling for future batches.")
                    train_models[model_name] = False
            except Exception as e:
                print(f"Error training {model_name} on this batch: {str(e)}")

        # Calculate current average accuracy across all models
        current_avg_acc = 0.0
        model_count = 0
        for model_name, accuracies in batch_accuracies.items():
            if accuracies:  # If there are any accuracy values
                current_avg_acc += np.mean(accuracies)
                model_count += 1

        if model_count > 0:
            current_avg_acc /= model_count

            # Check for early stopping
            if current_avg_acc > best_avg_acc:
                best_avg_acc = current_avg_acc
                no_improvement_count = 0
                # Save models when we get a new best accuracy
                print(f"\nNew best average accuracy: {best_avg_acc:.4f}. Saving models...")
                save_models_with_timestamp(models, tfidf, all_classes, batch_accuracies)
                print("Models saved successfully!")
            else:
                no_improvement_count += 1
                print(f"No improvement for {no_improvement_count} batches. (Best: {best_avg_acc:.4f}, Current: {current_avg_acc:.4f})")

                if no_improvement_count >= patience:
                    print(f"\nEarly stopping after {patience} batches without improvement.")
                    break

        # Periodic saving
        if batch_count % save_interval == 0:
            print(f"\nPeriodic save at batch {batch_count}...")
            save_models_with_timestamp(models, tfidf, all_classes, batch_accuracies)
            print("Models saved successfully!")

    # Print average batch accuracy for each model
    print("\nAverage Batch Accuracies:")
    for model_name, accuracies in batch_accuracies.items():
        avg_acc = np.mean(accuracies) if accuracies else 0.0
        print(f"{model_name}: {avg_acc:.4f}")

    # Final save of all models & TF-IDF Vectorizer with timestamp
    save_models_with_timestamp(models, tfidf, all_classes, batch_accuracies)
    print("\nAll models and vectorizer saved successfully!")

# Example prediction function
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

    Example:
        >>> prediction = predict_complaint("I have an issue with my credit card", "Logistic Regression")
        >>> print(prediction)
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

# Example usage
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
        "I tried to make a payment online, but it failed several times. Is there a problem with your system",
        "I’ve been charged interest on my credit card even though I paid on time. Please look into this issue"
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
