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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Linear SVC": LinearSVC(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),  # Full complexity
    "Multinomial NB": MultinomialNB(),
    "SGD Classifier": SGDClassifier(loss="hinge", random_state=42)  # Linear SVM-like classifier
}
batch_accuracies = {model_name: [] for model_name in models.keys()}  # Store batch accuracy
batch_precisions = {model_name: [] for model_name in models.keys()}  # Store batch precision
batch_recalls = {model_name: [] for model_name in models.keys()}  # Store batch recall
batch_f1_scores = {model_name: [] for model_name in models.keys()}  # Store batch F1-score

# Global variables for signal handling
tfidf_vectorizer = None
all_classes_list = None
batch_precisions = None
batch_recalls = None
batch_f1_scores = None

def save_models_with_timestamp(models_dict, tfidf, all_classes, batch_accuracies, batch_precisions=None, batch_recalls=None, batch_f1_scores=None):
    """
    Save all models with a timestamp in the filename.

    Args:
        models_dict (dict): Dictionary of model name to model object
        tfidf: TF-IDF vectorizer
        all_classes: List of unique classes
        batch_accuracies (dict): Dictionary of model name to list of batch accuracies
        batch_precisions (dict): Dictionary of model name to list of batch precisions
        batch_recalls (dict): Dictionary of model name to list of batch recalls
        batch_f1_scores (dict): Dictionary of model name to list of batch F1-scores
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
    global batch_precisions, batch_recalls, batch_f1_scores
    print("\nCtrl+C detected! Saving models before exiting...")
    save_models_with_timestamp(models, tfidf_vectorizer, all_classes_list, batch_accuracies, batch_precisions, batch_recalls, batch_f1_scores)
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
    global models, batch_accuracies, batch_precisions, batch_recalls, batch_f1_scores, tfidf_vectorizer, all_classes_list

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    # Load dataset in chunks
    dataset_path = "./datasets/complaints.csv"
    chunk_size = 512  # Adjust based on memory availability

    # Initialize TF-IDF Vectorizer (Fitted only once)
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
    tfidf_vectorizer = tfidf  # Update global variable

    # Update global variables for metrics tracking
    batch_precisions = {model_name: [] for model_name in models.keys()}
    batch_recalls = {model_name: [] for model_name in models.keys()}
    batch_f1_scores = {model_name: [] for model_name in models.keys()}

    # Initialize models for online learning


    # Flag to control which models to train (set to False to skip a model)
    train_models = {
        "Logistic Regression": True,
        "Linear SVC": True,
        "Random Forest": True,
        "Multinomial NB": True,
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

                # Calculate batch metrics
                batch_acc = accuracy_score(y_test, y_pred)
                batch_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                batch_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                batch_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                # Store batch metrics
                batch_accuracies[model_name].append(batch_acc)
                batch_precisions[model_name].append(batch_precision)
                batch_recalls[model_name].append(batch_recall)
                batch_f1_scores[model_name].append(batch_f1)

                training_time = time.time() - start_time
                print(f"{model_name} - Batch Metrics: Accuracy: {batch_acc:.4f}, Precision: {batch_precision:.4f}, Recall: {batch_recall:.4f}, F1-Score: {batch_f1:.4f}, Training Time: {training_time:.2f}s")

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
                save_models_with_timestamp(models, tfidf, all_classes, batch_accuracies, batch_precisions, batch_recalls, batch_f1_scores)
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
            save_models_with_timestamp(models, tfidf, all_classes, batch_accuracies, batch_precisions, batch_recalls, batch_f1_scores)
            print("Models saved successfully!")

    # Print average batch metrics for each model and track best performers
    print("\nAverage Batch Metrics:")

    # Dictionaries to track best models for each metric
    best_models = {
        'accuracy': {'model': None, 'value': 0},
        'precision': {'model': None, 'value': 0},
        'recall': {'model': None, 'value': 0},
        'f1': {'model': None, 'value': 0}
    }

    # Dictionary to store average metrics for each model
    model_metrics = {}

    for model_name in batch_accuracies.keys():
        accuracies = batch_accuracies[model_name]
        precisions = batch_precisions[model_name]
        recalls = batch_recalls[model_name]
        f1_scores = batch_f1_scores[model_name]

        if accuracies:  # If there are any metric values
            avg_acc = np.mean(accuracies)
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            avg_f1 = np.mean(f1_scores)

            # Store metrics for this model
            model_metrics[model_name] = {
                'accuracy': avg_acc,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1
            }

            # Update best models if this one is better
            if avg_acc > best_models['accuracy']['value']:
                best_models['accuracy'] = {'model': model_name, 'value': avg_acc}
            if avg_precision > best_models['precision']['value']:
                best_models['precision'] = {'model': model_name, 'value': avg_precision}
            if avg_recall > best_models['recall']['value']:
                best_models['recall'] = {'model': model_name, 'value': avg_recall}
            if avg_f1 > best_models['f1']['value']:
                best_models['f1'] = {'model': model_name, 'value': avg_f1}

            print(f"{model_name}:")
            print(f"  Accuracy: {avg_acc:.4f}")
            print(f"  Precision: {avg_precision:.4f}")
            print(f"  Recall: {avg_recall:.4f}")
            print(f"  F1-Score: {avg_f1:.4f}")
        else:
            print(f"{model_name}: No metrics available")

    # Print summary of best models for each metric
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*50)

    print("\nBest Model for Each Metric:")
    for metric, data in best_models.items():
        if data['model']:
            print(f"  Best {metric.capitalize()}: {data['model']} ({data['value']:.4f})")

    # Determine overall best model
    model_scores = {}
    for metric, data in best_models.items():
        if data['model']:
            if data['model'] not in model_scores:
                model_scores[data['model']] = 0
            model_scores[data['model']] += 1

    # Find model with highest score
    if model_scores:
        overall_best_model = max(model_scores.items(), key=lambda x: x[1])[0]

        # Calculate average rank across all metrics
        model_ranks = {model: [] for model in model_metrics.keys()}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            # Sort models by this metric
            sorted_models = sorted(model_metrics.items(), key=lambda x: x[1][metric], reverse=True)
            # Assign ranks
            for rank, (model, _) in enumerate(sorted_models, 1):
                model_ranks[model].append(rank)

        # Calculate average rank for each model
        avg_ranks = {model: np.mean(ranks) for model, ranks in model_ranks.items() if ranks}
        # Find model with best (lowest) average rank
        best_avg_rank_model = min(avg_ranks.items(), key=lambda x: x[1])[0]

        print(f"\nOverall Best Model (most wins): {overall_best_model}")
        print(f"Best Model by Average Rank: {best_avg_rank_model} (Avg Rank: {avg_ranks[best_avg_rank_model]:.2f})")

        print("\nDetailed Performance Comparison:")
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Avg Rank':<10}")
        print("-" * 70)
        for model, metrics in model_metrics.items():
            print(f"{model:<20} {metrics['accuracy']:.4f}     {metrics['precision']:.4f}     {metrics['recall']:.4f}     {metrics['f1']:.4f}     {avg_ranks.get(model, 'N/A'):<10}")
    else:
        print("\nNo models have been successfully trained and evaluated.")

    print("="*50)

    # Final save of all models & TF-IDF Vectorizer with timestamp
    save_models_with_timestamp(models, tfidf, all_classes, batch_accuracies, batch_precisions, batch_recalls, batch_f1_scores)
    print("\nAll models and vectorizer saved successfully!")

if __name__ == "__main__":
    train_model()
