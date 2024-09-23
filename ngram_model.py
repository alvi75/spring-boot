import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Function to calculate precision, recall, and F1 score
def RecallPrecisionFScore(y_test, pred):
    arr = confusion_matrix(y_test, pred)
    tp = np.diag(arr)
    fp = arr.sum(axis=0) - tp
    fn = arr.sum(axis=1) - tp

    precision = tp.sum() / (tp.sum() + fp.sum())
    recall = tp.sum() / (tp.sum() + fn.sum())
    f1 = (2 * precision * recall) / (precision + recall)

    print("Precision: {:.2f}".format(precision * 100))
    print("Recall: {:.2f}".format(recall * 100))
    print("F1 score: {:.2f}".format(f1 * 100))
    return precision * 100, recall * 100, f1 * 100

# Function to load tokenized methods from file
def load_tokenized_methods(file_path):
    with open(file_path, 'r') as f:
        methods = [line.strip() for line in f.readlines()]
    return methods

# Function for generating N-grams from lines of methods
def generate_ngrams(tokenized_methods, n):
    ngrams = []
    for i in range(len(tokenized_methods) - n + 1):
        ngram = ' '.join(tokenized_methods[i:i + n])
        ngrams.append(ngram)
    return ngrams

# Function for N-gram feature extraction and classification
def feature_based(tokenized_methods, labels, c, ngram_size):
    # Generate n-grams from lines of methods
    ngrams = generate_ngrams(tokenized_methods, ngram_size)
    
    # Adjust the labels to match the number of generated n-grams
    labels = labels[:len(ngrams)]

    # Vectorize the n-grams
    vectorizer = TfidfVectorizer(analyzer='word', tokenizer=lambda x: x.split(), stop_words=None)
    X = vectorizer.fit_transform(ngrams)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    # Train the model
    model = LinearSVC(C=c, penalty='l2', loss='squared_hinge')
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print classification report and confusion matrix
    print(classification_report(y_test, y_pred))
    precision, recall, f1 = RecallPrecisionFScore(y_test, y_pred)

# Main function
if __name__ == '__main__':
    # Load tokenized methods from file
    tokenized_methods = load_tokenized_methods("tokenized_methods.txt")

    # Create simulated labels (1 for half, 0 for the other half)
    labels = [1 if i < len(tokenized_methods) / 2 else 0 for i in range(len(tokenized_methods))]

    # Select N-gram models (Unigram, Bigram, Trigram, etc.)
    model_name = input("Please specify the N-gram model (Unigram, Bigram, Trigram, All): ")

    if model_name == "Unigram":
        feature_based(tokenized_methods, labels, c=1, ngram_size=1)

    elif model_name == "Bigram":
        feature_based(tokenized_methods, labels, c=10, ngram_size=2)

    elif model_name == "Trigram":
        feature_based(tokenized_methods, labels, c=1, ngram_size=3)

    elif model_name == "All":
        print("Running Unigram model...")
        feature_based(tokenized_methods, labels, c=1, ngram_size=1)
        print("\nRunning Bigram model...")
        feature_based(tokenized_methods, labels, c=10, ngram_size=2)
        print("\nRunning Trigram model...")
        feature_based(tokenized_methods, labels, c=1, ngram_size=3)
