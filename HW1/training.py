import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

def plot_confusion_matrix(y_true, y_pred):
    # Create a confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Create a heatmap
    sns.heatmap(cm ,annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    return 

def plot_classification_report(y_true, y_pred):
    # Create a classification report
    report = classification_report(y_true, y_pred, output_dict=True)

    metrics = ['precision', 'recall', 'f1-score']
    labels = ['0 (Negative)', '1 (Positive)']

    for metric in metrics:
        values = [report['0'][metrics], report['1'][metrics]]
        plt.bar(labels, values)
        plt.ylim(0, 1)
        plt.title(f'{metric.capitalize()} by Class')
        plt.ylabel(metric.capitalize())
        plt.show()

    plt.bar(['Accuracy'], [report['accuracy']])
    plt.ylim(0, 1)
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.show()
    return

def plot_roc_curve(model, x_test, y_test):
    # Get the predicted probabilities
    y_prob = model.predict_proba(x_test)[:, 1]
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    return 

def main():
    # Read the dataset (CSV) using pandas
    train_df = pd.read_csv('./HW1/train.csv', encoding='utf-8')
    test_df = pd.read_csv('./HW1/test.csv', encoding='utf-8')

    # Extract features and labels
    x_train = train_df['ClearedText']
    y_train = train_df['Sentiment']
    x_test = test_df['ClearedText']
    y_test = test_df['Sentiment']

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    # Fit and transform the training data  
    x_train_vector = vectorizer.fit_transform(x_train)
    
    # Transform the test data
    x_test_vector = vectorizer.transform(x_test)

    # Initialize the Multinomial Naive Bayes classifier
    classifier = MultinomialNB()
    # Train the classifier
    classifier.fit(x_train_vector, y_train)

    # Make predictions on the test data
    y_pred = classifier.predict(x_test_vector)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Evaluate and visualize the model
    # plot_confusion_matrix(y_test, y_pred)
    # plot_classification_report(y_test, y_pred)
    # plot_roc_curve(classifier, x_test_vector, y_test)

    return 

if __name__ == "__main__":
    main()