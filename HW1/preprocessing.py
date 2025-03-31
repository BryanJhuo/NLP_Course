from sklearn.model_selection import train_test_split
import pandas as pd
import re
import string
import nltk
# Uncomment the following line if you haven't downloaded the stopwords
def download_if_not_exists(resource_name: str):
    try:
        nltk.data.find(resource_name)
    except LookupError:
        nltk.download(resource_name)
    return 

download_if_not_exists("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = str(stopwords.words('english'))
stemmer = PorterStemmer()

# Path to the dataset
DATASET_PATH = './HW1/dataset.csv'

def clean_text(text: str) -> str:
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()  # Split into words
    words = [stemmer.stem(word) for word in words if word not in stop_words] # Remove stopwords and stem
    return "".join(words)  # Join words back into a string

def output_cleandataset(df: pd.DataFrame):
    # Save the cleaned dataset to a new CSV file
    df.to_csv('./HW1/cleaned_dataset.csv', index=False, encoding='utf-8')
    return 

def split_dataset(df: pd.DataFrame, test_size: float):
    # assign the x and y values
    x = df['ClearedText'] # features
    y = df['Sentiment'] # labels
    # split the dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, 
        y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )
    # create a new dataframe for train and test sets
    train_df = pd.DataFrame({'ClearedText': x_train, 'Sentiment': y_train})
    test_df = pd.DataFrame({'ClearedText': x_test, 'Sentiment': y_test})
    # save the train and test sets to csv files
    train_df.to_csv('./HW1/train.csv', index=False, encoding='utf-8')
    test_df.to_csv('./HW1/test.csv', index=False, encoding='utf-8')
    return 

def main():
    # read the dataset(csv) by pandas
    df = pd.read_csv(DATASET_PATH, encoding='latin1', quoting=1, quotechar='"', on_bad_lines='skip').head(200)
    
    # clean the text and store it in a new column
    df['ClearedText']= df["SentimentText"].apply(clean_text)

    # Create a new DataFrame with only the required columns
    new_df = df[["Sentiment", "ClearedText"]]
    
    # save the cleaned dataset
    output_cleandataset(new_df)

    # split the dataset into train and test sets
    split_dataset(new_df, 0.3)

    return 

if __name__ == "__main__":
    main()