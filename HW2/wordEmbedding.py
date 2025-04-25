import numpy as np
import csv
import os
from numpy.linalg import norm
from scipy.stats import spearmanr

GLOVE_PATH = ""  # Path to your GloVe file
WORDSIM_PATH = ""  # Path to your WordSim file
BATS_PATH = ""  # Path to your BATS file

def load_glove_embeddings(file_path):
    '''
    Load GloVe embeddings from a file.
    Args:
        file_path (str): Path to the GloVe file.
    Returns:
        dict: A dictionary mapping words to their GloVe vectors.
    '''
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f :
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def load_wordsim_data(file_path):
    '''
    Load WordSim dataset from a file.
    Args:
        file_path (str): Path to the WordSim file.
    Returns:
        list: A list of tuples containing word pairs and their similarity scores.
    '''
    word_pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            word1, word2, score = row[0].lower(), row[1].lower(), float(row[2])
            word_pairs.append((word1, word2, score))
    return word_pairs

def load_bats_data(file_path):
    '''
    Load BATS dataset from a file.
    Args:
        file_path (str): Path to the BATS file.
    Returns:
        list: A list of tuples containing word pairs and their similarity scores.
    '''
    analogy_questions = [] # list of (a, b, c, d) tuples
    for filename in os.listdir(file_path):
        if filename.endwith('.txt'):
            with open(os.path.join(file_path, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    words = line.strip().lower().split()
                    if len(words) == 2:
                        a, b = words
                        analogy_questions.append((a, b))            
    return analogy_questions

def cosine_similarity(vec1, vec2):
    '''
    Calculate cosine similarity between two vectors.
    Args:
        vec1 (np.array): First vector.
        vec2 (np.array): Second vector.
    Returns:
        float: Cosine similarity between the two vectors.
    '''
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def calculate_similarity(vectors, pairs):
    '''
    Calculate cosine similarity between word pairs and their gold standard scores.
    Args:
        vectors (dict): Dictionary of word vectors.
        pairs (list): List of tuples containing word pairs and their similarity scores.
    Returns:
        list: List of model similarity scores.
        list: List of gold similarity scores.
    '''
    model_scores = []
    gold_scores = []

    for word1, word2, score in pairs:
        vec1 = vectors.get(word1)
        vec2 = vectors.get(word2)

        if vec1 is not None and vec2 is not None:
            sim = cosine_similarity(vec1, vec2)
            print(f"Cosine similarity between {word1} and {word2}: {sim:.4f} (True score: {score})")
            model_scores.append(sim)
            gold_scores.append(score)
    
    return model_scores, gold_scores

def calculate_spearman(model_scores, gold_scores):
    '''
    Calculate Spearman correlation  between model scores and gold scores.
    Args:
        model_scores (list): List of model similarity scores.
        gold_scores (list): List of gold similarity scores.
    Returns:
        float: Spearman correlation coefficient.
    '''
    if len(model_scores) == 0 or len(gold_scores) == 0:
        return None
    if len(model_scores) != len(gold_scores):
        raise ValueError("Model scores and gold scores must have the same length.")
    correlation, _ = spearmanr(model_scores, gold_scores)
    return correlation

def main():
    word_vectors = load_glove_embeddings(GLOVE_PATH)
    word_pairs = load_wordsim_data(WORDSIM_PATH)
    model_scores, gold_scores = calculate_similarity(word_vectors, word_pairs)
    correlation = calculate_spearman(model_scores, gold_scores)

    analogy_pairs = load_bats_data(BATS_PATH)
    
    return 

if __name__ == "__main__":
    main()