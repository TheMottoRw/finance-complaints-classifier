import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Load dataset in chunks
dataset_path = "./datasets/complaints.csv"  # Update with actual dataset path
chunk_size = 512  # Adjust based on memory availability
tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))

# Initialize model for online learning
model = SGDClassifier(loss="hinge")  # Linear SVM-like classifier that supports large dataset
first_batch = True  # Track if it's the first batch for fitting classes

# Store accuracy results
batch_accuracies = []
# Read dataset in chunks
reader = pd.read_csv(dataset_path, chunksize=chunk_size)

for chunk in reader:
    if chunk.shape[0] ==0:
        break
    df1 = chunk[['Product', 'Consumer complaint narrative']].copy()
    # Remove missing values (NaN)
    df1 = df1[pd.notnull(df1['Consumer complaint narrative'])]
    # Renaming second column for a simpler name
    # print(df.columns)
    df1.columns = ['Product', 'Consumer_complaint']
    # print(df1.shape)
    # print(df1.head(3).T)
    df1_values = pd.DataFrame(df1.Product.unique()).values

    # chunk.dropna(subset=["Consumer Complaint", "Product"], inplace=True)
    # chunk["Product"] = chunk["Product"].astype("category").cat.codes  # Encode labels

    # Split batch into train and test
    X = df1['Consumer_complaint'] # Collection of documents
    y = df1['Product'] # Target or the labels we want to predict (i.e., the 15 different complaints of products)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                   random_state = 0)
    # X_train, X_test, y_train, y_test = train_test_split(chunk["Consumer Complaint"], chunk["Product"], test_size=0.2, stratify=chunk["Product"])

    # Convert text to TF-IDF features
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train incrementally
    if first_batch:
        model.partial_fit(X_train_tfidf, y_train, classes=np.unique(y_train))
        first_batch = False
    else:
        model.partial_fit(X_train_tfidf, y_train)

    # Predict batch
    y_pred = model.predict(X_test_tfidf)

    # Calculate batch accuracy
    batch_acc = accuracy_score(y_test, y_pred)
    batch_accuracies.append(batch_acc)
    print(f"Batch Accuracy: {batch_acc:.4f}")
print(f"Average Batch Accuracy: {np.mean(batch_accuracies):.4f}")
