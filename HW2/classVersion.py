import numpy as np
import multiprocessing as mp
import csv
import os
from numpy.linalg import norm
from scipy.stats import spearmanr
from collections import defaultdict
from tqdm import tqdm
    

GLOVE_PATH = "HW2/glove.6B.100d.txt"  # Path to your GloVe file
WORDSIM_PATH = "HW2/combined.csv"  # Path to your WordSim file
BATS_PATH = "HW2/BATS_3.0"  # Path to your BATS file

class WordEmbedding: 
    def __init__(self, glove_path, wordsim_path, bats_path):
        self.glove_path = glove_path
        self.wordsim_path = wordsim_path
        self.bats_path = bats_path
        self.word_vectors = {}
        self.vocab = []
        self.vecs = None
        self.word2idx = {}

    def load_glove_embeddings(self, file_path):
        embeddings = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                embeddings[word] = vector
        return embeddings

    def load_wordsim_data(self, file_path):
        word_pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                word1, word2, score = row[0].lower(), row[1].lower(), float(row[2])
                word_pairs.append((word1, word2, score))
        return word_pairs

    def load_bats_data(self, file_path):
        analogy_questions = []
        for filename in os.listdir(file_path):
            if filename.endswith('.txt'):
                with open(os.path.join(file_path, filename), 'r', encoding='utf-8') as f:
                    lines = [line.strip().lower().split() for line in f if len(line.strip().split()) == 2]
                    for i in range(len(lines)):
                        for j in range(len(lines)):
                            if i != j:
                                a, b = lines[i]
                                c, d = lines[j]
                                analogy_questions.append((a, b, c, d))
        return analogy_questions

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    def calculate_similarity(self, vectors, pairs):
        model_scores = []
        gold_scores = []

        for word1, word2, score in pairs:
            vec1 = vectors.get(word1)
            vec2 = vectors.get(word2)

            if vec1 is not None and vec2 is not None:
                sim = self.cosine_similarity(vec1, vec2)
                model_scores.append(sim)
                gold_scores.append(score)

        return model_scores, gold_scores

    def calculate_spearman(self, model_scores, gold_scores):
        if len(model_scores) == 0 or len(gold_scores) == 0:
            return None
        if len(model_scores) != len(gold_scores):
            raise ValueError("Model scores and gold scores must have the same length.")
        correlation, _ = spearmanr(model_scores, gold_scores)
        return correlation

    def predict_analogy(self, a, b, c, word_vectors):
        if a not in word_vectors or b not in word_vectors or c not in word_vectors:
            return None

        target_vector = word_vectors[b] - word_vectors[a] + word_vectors[c]

        max_sim = -1
        best_word = None
        for word, vector in word_vectors.items():
            if word in [a, b, c]:
                continue
            sim = self.cosine_similarity(target_vector, vector)
            if sim > max_sim:
                max_sim = sim
                best_word = word
        return best_word

    def predict_analogy_top_n(self, analogy_pairs, word_vectors, top_list=[1, 3, 5]):
        total = 0
        correct_at_k = defaultdict(int)

        for (a, b, c, d) in analogy_pairs:
            if a not in word_vectors or b not in word_vectors or c not in word_vectors:
                continue
            target_vector = word_vectors[b] - word_vectors[a] + word_vectors[c]

            sims = []
            for word, vector in word_vectors.items():
                if word in [a, b, c]:
                    continue
                sim = self.cosine_similarity(target_vector, vector)
                sims.append((word, sim))

            sims.sort(key=lambda x: x[1], reverse=True)
            total += 1

            for k in top_list:
                topk_words = [word for word, _ in sims[:k]]
                if d in topk_words:
                    correct_at_k[k] += 1

        accuracy_at_k = {k: (correct_at_k[k] / total) if total > 0 else 0.0 for k in top_list}
        return accuracy_at_k

    def predict_analogy_accuracy(self, analogy_pairs, word_vectors):
        correct, total = 0, 0
        for (a, b, c, d) in analogy_pairs:
            predicted = self.predict_analogy(a, b, c, word_vectors)
            if predicted == d:
                correct += 1
            total += 1
        if total == 0:
            return 0.0
        return correct / total

    def predict_analogy_worker(self, sub_pairs, vocab, vec, word2idx, top_list, return_dict, proc_id):
        correct_at_k = defaultdict(int)
        total = 0

        pbar = tqdm(total=len(sub_pairs), position=proc_id, desc=f"Process {proc_id}", leave=True)

        for (a, b, c, d) in sub_pairs:
            if a not in word2idx or b not in word2idx or c not in word2idx:
                pbar.update(1)
                continue

            vector_a = vec[word2idx[a]]
            vector_b = vec[word2idx[b]]
            vector_c = vec[word2idx[c]]
            target_vector = vector_b - vector_a + vector_c

            sims = np.dot(vec, target_vector) / (norm(vec, axis=1) * norm(target_vector))

            top_indices = np.argsort(-sims)

            total += 1
            for k in top_list:
                topk_words = [vocab[i] for i in top_indices[:k]]
                if d in topk_words:
                    correct_at_k[k] += 1

            pbar.update(1)

        pbar.close()
        return_dict[proc_id] = (correct_at_k, total)

    def predict_analogy_top_k_parallel(self, analogy_pairs, vocab, vec, word2idx, top_list=[1, 3, 5], num_workers=None):
        if num_workers is None:
            num_workers = mp.cpu_count()

        manager = mp.Manager()
        return_dict = manager.dict()
        jobs = []

        chunk_size = len(analogy_pairs) // num_workers
        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = None if i == num_workers - 1 else (i + 1) * chunk_size
            sub_pairs = analogy_pairs[start_idx:end_idx]
            p = mp.Process(target=self.predict_analogy_worker, args=(sub_pairs, vocab, vec, word2idx, top_list, return_dict, i))
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()

        total = 0
        correct_at_k = defaultdict(int)
        for _, (sub_correct, sub_total) in return_dict.items():
            total += sub_total
            for k in top_list:
                correct_at_k[k] += sub_correct[k]

        accuracy_at_k = {k: (correct_at_k[k] / total) if total > 0 else 0.0 for k in top_list}
        return accuracy_at_k
    
    def run_task1_load_embeddings(self):
        print("[Task 1] Loading GloVe embeddings...")
        self.word_vectors = self.load_glove_embeddings(self.glove_path)
        self.vocab = list(self.word_vectors.keys())
        self.vecs = np.stack([self.word_vectors[word] for word in self.vocab])
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        print(f"Loaded {len(self.word_vectors)} word vectors.")
        print(f"Shape of word vectors: {self.vecs.shape}")

    def run_task2_wordsim_similarity(self):
        print("\n[Task 2] Loading WordSim353 data and testing similarity...")
        word_pairs = self.load_wordsim_data(self.wordsim_path)
        model_scores, gold_scores = self.calculate_similarity(self.word_vectors, word_pairs)
        spearman_corr = self.calculate_spearman(model_scores, gold_scores)
        if spearman_corr is not None:
            print(f"Spearman correlation: {spearman_corr:.4f}")
        else:
            print("No valid pairs for Spearman correlation calculation.")

    def run_task3_bats_analogy(self, num_workers=6):
        print("\n[Task 3] Loading BATS data and testing analogy predictions (Top-k & parallel)...")
        analogy_pairs = self.load_bats_data(self.bats_path)
        acc_result = self.predict_analogy_top_k_parallel(analogy_pairs, self.vocab, self.vecs, self.word2idx, top_list=[1, 3, 5], num_workers=num_workers)
        for k in [1, 3, 5]:
            print(f"\nTop-{k} accuracy: {acc_result[k]:.10f}")

def main():
    model = WordEmbedding(GLOVE_PATH, WORDSIM_PATH, BATS_PATH)
    model.run_task1_load_embeddings()
    model.run_task2_wordsim_similarity()
    model.run_task3_bats_analogy(num_workers=6)
    return 

if __name__ == "__main__":
    main()
