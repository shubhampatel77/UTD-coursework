from collections import Counter, defaultdict
import math
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class NgramModel:
    def __init__(self, n=2, unk_strategy='frequency', unk_threshold=1, vocab_size=None):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.vocab = set()
        self.total_words = 0
        self.unk_strategy = unk_strategy
        self.unk_threshold = unk_threshold
        self.vocab_size = vocab_size

    def preprocess(self, text):
        # Convert to lowercase and tokenize
        tokens = text.lower().split()
        return tokens

    def handle_unknown_words(self, tokens):
        if self.unk_strategy == 'frequency':
            return [word if self.word_counts[word] > self.unk_threshold else '<UNK>' for word in tokens]
        elif self.unk_strategy == 'vocab_size':
            most_common = set(word for word, _ in self.word_counts.most_common(self.vocab_size))
            return [word if word in most_common else '<UNK>' for word in tokens]

    def train(self, corpus):
        # First pass: count words for UNK replacement
        self.word_counts = Counter()
        for review in tqdm(corpus, desc="Counting words"):
            tokens = self.preprocess(review)
            self.word_counts.update(tokens)

        # Second pass: actual training
        for review in tqdm(corpus, desc="Training model"):
            tokens = self.preprocess(review)
            tokens = self.handle_unknown_words(tokens)
            self.total_words += len(tokens)
            self.vocab.update(tokens)

            padded_tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
            for i in range(len(padded_tokens) - self.n + 1):
                ngram = tuple(padded_tokens[i:i + self.n])
                prefix = ngram[:-1]
                word = ngram[-1]
                self.ngram_counts[prefix][word] += 1

    def add_k_smoothing(self, prefix, word, k=1):
        prefix_count = sum(self.ngram_counts[prefix].values())
        vocab_size = len(self.vocab)
        word_count = self.ngram_counts[prefix][word]
        probability = (word_count + k) / (prefix_count + k * vocab_size)
        return probability

    def linear_interpolation(self, prefix, word, lambda_param=0.5, k=1):
        bigram_prob = self.add_k_smoothing(prefix, word, k)
        unigram_prob = (self.word_counts[word] + k) / (self.total_words + k * len(self.vocab))
        probability = lambda_param * bigram_prob + (1 - lambda_param) * unigram_prob
        return probability

    def unsmoothed_unigram_probability(self, word):
        word_count = self.word_counts[word]
        return word_count / self.total_words

    def unsmoothed_probability(self, prefix, word):
        prefix_count = sum(self.ngram_counts[prefix].values())
        word_count = self.ngram_counts[prefix][word]
        if prefix_count == 0:
            return 0
        else:
            return word_count / prefix_count

    def perplexity(self, test_corpus, smoothing_method='add_k', **kwargs):
        log_prob_sum = 0
        total_words = 0

        for review in tqdm(test_corpus, desc=f"Calculating perplexity ({smoothing_method})"):
            tokens = self.preprocess(review)
            tokens = self.handle_unknown_words(tokens)
            total_words += len(tokens)

            padded_tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
            for i in range(self.n - 1, len(padded_tokens)):
                prefix = tuple(padded_tokens[i - self.n + 1:i])
                word = padded_tokens[i]

                if smoothing_method == 'add_k':
                    prob = self.add_k_smoothing(prefix, word, kwargs.get('k', 1))
                elif smoothing_method == 'linear_interpolation':
                    prob = self.linear_interpolation(prefix, word, kwargs.get('lambda_param', 0.5))
                else:
                    raise ValueError("Invalid smoothing method")

                # if prob > 0:
                log_prob_sum += -math.log(prob)
                # else:
                #     # Assign a large negative log probability for zero probabilities
                #     log_prob_sum += float('inf')

        perplexity = math.exp(log_prob_sum / total_words)
        return perplexity

def split_data(data, train_ratio=0.8):
    random.seed(42)
    random.shuffle(data)
    split_point = int(len(data) * train_ratio)
    return data[:split_point], data[split_point:]

def plot_hyperparameter_tuning(hyperparameter_results):
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Hyperparameter Tuning Results', fontsize=16)

    # Add-k smoothing
    ax = axes[0, 0]
    for unk_strategy in ['frequency', 'vocab_size']:
        vocab_sizes = [None] if unk_strategy == 'frequency' else [1000, 2500, 5000, 6380]
        for vocab_size in vocab_sizes:
            k_values = [0.1, 0.5, 1.0, 2.0, 5.0]
            perplexities = []
            for k in k_values:
                key = (unk_strategy, 'add_k', vocab_size, 'k', k)
                perplexity = hyperparameter_results.get(key, None)
                if perplexity is not None:
                    perplexities.append(perplexity)
                else:
                    perplexities.append(float('inf'))
            label = f"{unk_strategy}" if vocab_size is None else f"{unk_strategy}_{vocab_size}"
            ax.plot(k_values, perplexities, marker='o', label=label)
    ax.set_xlabel('k value')
    ax.set_ylabel('Perplexity')
    ax.set_title('Add-k Smoothing')
    ax.legend()
    ax.set_yscale('log')

    # Linear Interpolation
    ax = axes[0, 1]
    for unk_strategy in ['frequency', 'vocab_size']:
        vocab_sizes = [None] if unk_strategy == 'frequency' else [1000, 2500, 5000, 6380]
        for vocab_size in vocab_sizes:
            lambda_values = [0.1, 0.3, 0.5, 0.7, 0.9]
            perplexities = []
            for lambda_param in lambda_values:
                key = (unk_strategy, 'linear_interpolation', vocab_size, 'lambda', lambda_param)
                perplexity = hyperparameter_results.get(key, None)
                if perplexity is not None:
                    perplexities.append(perplexity)
                else:
                    perplexities.append(float('inf'))
                # print(f"vocab_size = {vocab_size}, perplexities = {perplexities}")
            label = f"{unk_strategy}" if vocab_size is None else f"{unk_strategy}_{vocab_size}"
            ax.plot(lambda_values, perplexities, marker='o', label=label)
    ax.set_xlabel('Lambda value')
    ax.set_ylabel('Perplexity')
    ax.set_title('Linear Interpolation')
    ax.legend()
    ax.set_yscale('log')

    # UNK strategies comparison
    ax = axes[1, 0]
    unk_labels = []
    best_perplexities = []
    for unk_strategy in ['frequency', 'vocab_size']:
        vocab_sizes = [None] if unk_strategy == 'frequency' else [1000, 2500, 5000, int(0.8*6380)]
        for vocab_size in vocab_sizes:
            label = f"{unk_strategy}" if vocab_size is None else f"{unk_strategy}_{vocab_size}"
            unk_labels.append(label)
            perplexity_list = []
            for key in hyperparameter_results.keys():
                if key[0] == unk_strategy and key[2] == vocab_size:
                    perplexity_list.append(hyperparameter_results[key])
            best_perplexities.append(min(perplexity_list) if perplexity_list else float('inf'))
    ax.bar(unk_labels, best_perplexities)
    ax.set_xlabel('UNK Strategy')
    ax.set_ylabel('Best Perplexity')
    ax.set_title('Best Perplexity by UNK Strategy')
    ax.set_yscale('log')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Smoothing methods comparison
    ax = axes[1, 1]
    smoothing_methods = ['add_k', 'linear_interpolation']
    method_labels = []
    best_perplexities = []
    for method in smoothing_methods:
        method_labels.append(method)
        perplexity_list = []
        for key in hyperparameter_results.keys():
            if key[1] == method:
                perplexity_list.append(hyperparameter_results[key])
        best_perplexities.append(min(perplexity_list) if perplexity_list else float('inf'))
    ax.bar(method_labels, best_perplexities)
    ax.set_xlabel('Smoothing Method')
    ax.set_ylabel('Best Perplexity')
    ax.set_title('Best Perplexity by Smoothing Method')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('hyperparameter_tuning_results.png')
    plt.close()
    
    
def plot_learning_curves(train_data, val_data, unk_strategy, smoothing_method, **kwargs):
    train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    train_perplexities = []
    val_perplexities = []

    for size in train_sizes:
        subset_size = int(len(train_data) * size)
        subset_train_data = train_data[:subset_size]
        
        model = NgramModel(n=2, unk_strategy=unk_strategy, vocab_size=kwargs.get('vocab_size'))
        model.train(subset_train_data)
        
        train_perplexity = model.perplexity(subset_train_data, smoothing_method, **kwargs)
        val_perplexity = model.perplexity(val_data, smoothing_method, **kwargs)
        
        train_perplexities.append(train_perplexity)
        val_perplexities.append(val_perplexity)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_perplexities, label='Training Perplexity')
    plt.plot(train_sizes, val_perplexities, label='Validation Perplexity')
    plt.xlabel('Training Set Size')
    plt.ylabel('Perplexity')
    plt.title(f'Learning Curves ({unk_strategy}, {smoothing_method})')
    plt.legend()
    plt.savefig(f'learning_curves_{unk_strategy}_{smoothing_method}.png')
    plt.close()
    
def error_analysis(model, test_data, n=10):
    errors = []
    for review in test_data:
        tokens = model.preprocess(review)
        tokens = model.handle_unknown_words(tokens)
        padded_tokens = ['<s>'] * (model.n - 1) + tokens + ['</s>']
        
        for i in range(model.n - 1, len(padded_tokens) - 1):
            prefix = tuple(padded_tokens[i - model.n + 1:i])
            actual_word = padded_tokens[i]
            next_word = padded_tokens[i + 1]
            
            # Get top prediction
            top_prediction = max(model.ngram_counts[prefix], key=model.ngram_counts[prefix].get)
            
            # Calculate probabilities
            actual_prob = model.add_k_smoothing(prefix, actual_word, k=.1)
            pred_prob = model.add_k_smoothing(prefix, top_prediction, k=.1)
            
            error = abs(actual_prob - pred_prob)
            
            # Check if words are in top 1000 vocabulary
            in_top_vocab = lambda w: w in set(word for word, _ in model.word_counts.most_common(1000))
            
            errors.append((prefix, actual_word, top_prediction, error, in_top_vocab(actual_word), in_top_vocab(top_prediction)))
    
    top_errors = sorted(errors, key=lambda x: x[3], reverse=True)[:n]
    return top_errors


def main():
    # Adjust paths
    with open('../data/train.txt', 'r') as f:
        all_train_data = f.readlines()

    with open('../data/validation.txt', 'r') as f:
        validation_data = f.readlines()

    # Step 1: Initial approach without internal validation
    print("Step 1: Initial approach without internal validation")
    initial_model = NgramModel(n=2, unk_strategy='frequency')
    initial_model.train(all_train_data)
    initial_perplexity = initial_model.perplexity(validation_data, 'add_k', k=1)
    print(f"Initial perplexity on validation set: {initial_perplexity}")

    # Step 2: Analyze vocabulary and word frequency
    total_word_counts = Counter()
    for review in all_train_data:
        tokens = review.lower().split()
        total_word_counts.update(tokens)
    total_vocab_size = len(total_word_counts)
    print(f"Total vocabulary size in training data: {total_vocab_size}")
    
    word_frequencies = [count for word, count in total_word_counts.most_common()]
    plt.figure(figsize=(10, 6))
    plt.loglog(word_frequencies)
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Word Frequency Distribution')
    plt.savefig('word_frequency_distribution.png')
    plt.close()
    
    # Function to calculate perplexity for all configurations
    def calculate_perplexities(model, data, name):
        results = []
        for smoothing_method in ['add_k', 'linear_interpolation']:
            if smoothing_method == 'add_k':
                for k in [0.1, 0.5, 1.0, 2.0, 5.0]:
                    perplexity = model.perplexity(data, smoothing_method, k=k)
                    results.append(f"{name},{smoothing_method},k={k},{perplexity}")
            else:
                for lambda_param in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    perplexity = model.perplexity(data, smoothing_method, lambda_param=lambda_param)
                    results.append(f"{name},{smoothing_method},lambda={lambda_param},{perplexity}")
        return results

    # Step 3: Implement internal validation
    print("\nStep 3: Implementing internal validation")
    train_data, internal_val_data = split_data(all_train_data, 0.8)

    # Selected combinations based on previous results
    selected_combinations = [
        ('frequency', None, 'add_k', 0.1),
        ('frequency', None, 'add_k', 1.0),
        ('frequency', None, 'linear_interpolation', 0.5),
        ('vocab_size', 1000, 'add_k', 0.1),
        ('vocab_size', 1000, 'linear_interpolation', 0.5),
        ('vocab_size', 2500, 'add_k', 0.1),
        ('vocab_size', 5000, 'add_k', 0.1),
        ('vocab_size', 5000, 'linear_interpolation', 0.5),
        ('vocab_size', int(0.8*6380), 'add_k', 0.1),
        ('vocab_size', int(0.8*6380), 'linear_interpolation', 0.5)
    ]

    all_results = []

    # Step 4: Comprehensive perplexity calculation
    # Ignore this step to prevent lot of computation
    
    # print("Step 4: Calculating perplexities for selected combinations")
    # for unk_strategy, vocab_size, smoothing_method, param in selected_combinations:
    #     model = NgramModel(n=2, unk_strategy=unk_strategy, vocab_size=vocab_size)
        
    #     # Train on 80% of data
    #     model.train(train_data)
    #     all_results.extend(calculate_perplexities(model, train_data, f"train_80%_{unk_strategy}_{vocab_size}"))
    #     all_results.extend(calculate_perplexities(model, internal_val_data, f"internal_val_{unk_strategy}_{vocab_size}"))
        
    #     model = NgramModel(n=2, unk_strategy=unk_strategy, vocab_size=vocab_size)
    #     # Train on full data
    #     model.train(all_train_data)
    #     all_results.extend(calculate_perplexities(model, all_train_data, f"train_full_{unk_strategy}_{vocab_size}"))
    #     all_results.extend(calculate_perplexities(model, validation_data, f"validation_{unk_strategy}_{vocab_size}"))

    # # Write results to file
    # with open('perplexity_results.csv', 'w') as f:
    #     f.write("dataset,smoothing_method,parameter,perplexity\n")
    #     for result in all_results:
    #         f.write(f"{result}\n")

    # Step 5: Find best hyperparameters
    best_perplexity = float('inf')
    best_params = {}
    for result in all_results:
        if result.startswith("internal_val"):
            _, smoothing_method, param, perplexity = result.split(',')
            perplexity = float(perplexity)
            if perplexity < best_perplexity:
                best_perplexity = perplexity
                best_params = {
                    'smoothing_method': smoothing_method,
                    param.split('=')[0]: float(param.split('=')[1])
                }

    print(f"Best hyperparameters: {best_params}")
    print(f"Best perplexity on internal validation set: {best_perplexity}")

    # Step 6: Train final model and evaluate
    print("\nStep 6: Training final model and evaluating")
    final_model = NgramModel(n=2, unk_strategy='vocab_size', vocab_size=1000)  # Using best UNK strategy from previous results
    final_model.train(all_train_data)

    final_perplexity = final_model.perplexity(validation_data, best_params['smoothing_method'], 
                                              **{k: v for k, v in best_params.items() if k != 'smoothing_method'})
    print(f"Final perplexity on validation set: {final_perplexity}")

    # Step 7: Error analysis
    print("\nStep 7: Error analysis")
    top_errors = error_analysis(final_model, validation_data)
    print("Top 10 errors:")
    for prefix, actual, predicted, error, actual_in_top, pred_in_top in top_errors:
        print(f"Prefix: {prefix}, Actual: {actual} ({'in' if actual_in_top else 'not in'} top 1000), "
              f"Predicted: {predicted} ({'in' if pred_in_top else 'not in'} top 1000), Error: {error:.4f}")

if __name__ == "__main__":
    main()