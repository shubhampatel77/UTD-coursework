import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm.auto import tqdm
import json
from argparse import ArgumentParser

# Added imports
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h, dropout_rate=0.2):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.dropout = nn.Dropout(dropout_rate)  # Add dropout for regularization
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        hidden = self.activation(self.W1(input_vector))
        # [to fill] obtain output layer representation
        output = self.W2(hidden)
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output)

        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in tqdm(data, desc="Vectorizing"):
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data, test_data):
    with open(train_data) as f:
        training = json.load(f)
    with open(val_data) as f:
        validation = json.load(f)
    with open(test_data) as f:
        testing = json.load(f)

    train = []
    val = []
    test = []
    for x in training:
        train.append((x["text"].split(), int(x["stars"]-1)))
    for x in validation:
        val.append((x["text"].split(), int(x["stars"]-1)))
    for x in testing:
        test.append((x["text"].split(), int(x["stars"]-1)))

    return train, val, test
def training_loop(model, optimizer, device, train_data, epoch):
    model.train()
    train_loss = None
    train_correct = 0
    train_total = 0
    start_time = time.time()
    print("Training started for epoch {}".format(epoch + 1))
    random.shuffle(train_data) # Good practice to shuffle order of training data
    minibatch_size = 16 
    N = len(train_data)
    
    epoch_train_loss = 0.0  # Track total training loss for the epoch
    
    for minibatch_index in tqdm(range(N // minibatch_size), desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        batch_loss = None
        for example_index in range(minibatch_size):
            input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
            predicted_vector = model(input_vector.to(device))
            predicted_label = torch.argmax(predicted_vector)
            train_correct += int(predicted_label == gold_label)
            train_total += 1
            example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]).to(device))
            if batch_loss is None:
                batch_loss = example_loss
            else:
                batch_loss += example_loss
        batch_loss = batch_loss / minibatch_size
        batch_loss.backward()
        optimizer.step()
        
        epoch_train_loss += batch_loss.item() * minibatch_size
    # Add: avg trining loss over all examples, not avg of minibatch loss
    avg_train_loss = epoch_train_loss / N
    train_accuracy = train_correct / train_total
    
    print("Training accuracy for epoch {}: {}".format(epoch + 1, train_accuracy))
    print("Training loss for epoch {}: {}".format(epoch + 1, avg_train_loss))
    print("Training time for this epoch: {}".format(time.time() - start_time))
    
    return avg_train_loss, train_accuracy

def validation_loop(model, device, data, epoch, testing=False): 
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    minibatch_size = 16 
    N = len(data)
    
    epoch_loss = 0.0  # Track total validation loss for the epoch
    num_batches = 0
    
    with torch.no_grad():  # Add: No need to track gradients for validation
        for minibatch_index in tqdm(range(N // minibatch_size), desc=f"Eval {epoch+1}"):
            batch_loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector.to(device))
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]).to(device))
                if batch_loss is None:
                    batch_loss = example_loss
                else:
                    batch_loss += example_loss
            batch_loss = batch_loss / minibatch_size
            epoch_loss += batch_loss.item()
            num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    accuracy = correct / total
    
    dataset_name = "Validation" if not testing else "Testing"
    print(f"{dataset_name} accuracy for epoch {epoch+1}: {accuracy}")
    print(f"{dataset_name} loss for epoch {epoch + 1}: {avg_loss}")
    print(f"{dataset_name} time for this epoch: {time.time() - start_time}")
    
    return avg_loss, accuracy

# Add save metrics utility function
def save_metrics(model, epoch_metrics, base_output_dir, use_stratified):
    """Save training metrics and plots"""
    dataset_name = "stratified" if use_stratified else "original"
    output_dir = os.path.join(base_output_dir, dataset_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(epoch_metrics)
    metrics_df.to_csv(f"{output_dir}/training_metrics.csv", index=False)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train')
    plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Validation')
    plt.title('Loss vs Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics_df['epoch'], metrics_df['train_acc'], label='Train')
    plt.plot(metrics_df['epoch'], metrics_df['val_acc'], label='Validation')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png")
    plt.close()
    
# Add analyze errors utility function
def analyze_errors(model, data_loader, word2index, index2word, output_dir):
    """Analyze model errors and generate confusion matrix"""
    all_preds = []
    all_labels = []
    error_examples = []
    
    model.eval()
    with torch.no_grad():
        for input_vector, gold_label in tqdm(data_loader, desc="Analyzing errors"):
            pred = model(input_vector.to(device))
            pred_label = torch.argmax(pred).item()
            
            if pred_label != gold_label:
                # Convert vector back to words for error analysis
                words = [index2word[idx] for idx, count in enumerate(input_vector) if count > 0]
                error_examples.append({
                    'text': ' '.join(words),
                    'predicted': pred_label,
                    'actual': gold_label
                })
            
            all_preds.append(pred_label)
            all_labels.append(gold_label)
            
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()
    
    # Save error examples
    pd.DataFrame(error_examples).to_csv(f"{output_dir}/error_examples.csv", index=False)
    
def evaluation(model, data_dict, word2index, index2word, output_dir):
    
    analyze_errors(model, data_dict['test_report'], word2index, index2word, output_dir)
    # Generate classification reports
    for report_name in data_dict.keys():
        predicted_labels = []
        true_labels = []
        model.eval()
        with torch.no_grad():
            for input_vector, gold_label in data_dict[report_name]:
                pred = model(input_vector.to(device))
                predicted_labels.append(torch.argmax(pred).item())
                true_labels.append(gold_label)
        
        report = classification_report(true_labels, predicted_labels)
        with open(f"{output_dir}/{report_name}.txt", "w") as f:
            f.write(report)

def grid_search_hyperparameters(vocab, device, train_data, val_data, base_output_dir, use_stratified):
    """
    Perform grid search over hyperparameters
    Returns: List of dicts containing results for each configuration
    """
    # Define parameter grid
    param_grid = {
        'hidden_dim': [64, 128, 192],
        'dropout': [0.1, 0.2, 0.4],
        'learning_rate': [0.01, 0.005],
    }
    
    results = []
    
    # Generate all combinations
    for hidden_dim in param_grid['hidden_dim']:
        for dropout in param_grid['dropout']:
            for lr in param_grid['learning_rate']:
                # Create unique experiment name
                exp_name = f"hd{hidden_dim}_dr{dropout}_lr{lr}"
                dataset_name = "stratified" if use_stratified else "original"
                exp_dir = os.path.join(base_output_dir, "ffnn", dataset_name, exp_name)
                os.makedirs(exp_dir, exist_ok=True)
                
                print(f"\nStarting experiment: {exp_name}")
                print("=" * 50)
                print(f"Hidden dim: {hidden_dim}")
                print(f"Dropout: {dropout}")
                print(f"Learning rate: {lr}")
                
                # Initialize model and optimizer
                model = FFNN(input_dim=len(vocab), h=hidden_dim, dropout_rate=dropout).to(device)
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                
                # Training loop with early stopping
                best_val_acc = 0
                best_epoch = 0
                patience = 5
                no_improve_count = 0
                epoch_metrics = []
                
                for epoch in range(20):  # Max epochs
                    model.train()
                    train_loss, train_acc = training_loop(model, optimizer, device, train_data, epoch)
                    
                    val_loss, val_acc = validation_loop(model, device, val_data, epoch)
                    
                    # Save epoch metrics
                    metrics = {
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'hidden_dim': hidden_dim,
                        'dropout': dropout,
                        'learning_rate': lr,
                        'stratified': use_stratified
                    }
                    epoch_metrics.append(metrics)
                    
                    # Save epoch metrics to CSV
                    pd.DataFrame(epoch_metrics).to_csv(
                        os.path.join(exp_dir, 'metrics.csv'), index=False
                    )
                    
                    # Early stopping check
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_epoch = epoch
                        no_improve_count = 0
                        # Save best model
                        torch.save(model.state_dict(), 
                                    os.path.join(exp_dir, 'best_model.pt'))
                    else:
                        no_improve_count += 1
                        
                    if no_improve_count >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                
                # Save final results
                results.append({
                    'hidden_dim': hidden_dim,
                    'dropout': dropout,
                    'learning_rate': lr,
                    'best_val_acc': best_val_acc,
                    'best_epoch': best_epoch,
                    'exp_name': exp_name,
                    'stratified': use_stratified
                })
                
                # Save results summary
                pd.DataFrame(results).to_csv(
                    os.path.join(base_output_dir, 'grid_search_results.csv'), 
                    index=False
                )
    
    return results

def combine_and_stratify_data(train_data, val_data, test_data=None):
    """
    Combines and redistributes data while maintaining original ratios
    """
    # Combine all data
    all_data = train_data.copy()
    all_data.extend(val_data)
    if test_data:
        all_data.extend(test_data)
    
    total_samples = len(all_data)
    train_ratio = len(train_data) / total_samples
    val_ratio = len(val_data) / total_samples
    
    print("\nOriginal distribution:")
    print(f"Train: {len(train_data)} samples ({train_ratio:.2%})")
    print(f"Val: {len(val_data)} samples ({val_ratio:.2%})")
    if test_data:
        test_ratio = len(test_data) / total_samples
        print(f"Test: {len(test_data)} samples ({test_ratio:.2%})")
    
    # Create stratified split maintaining original ratios
    stratified_train, stratified_val, stratified_test = create_stratified_split(
        all_data, 
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=1-train_ratio-val_ratio if test_data else 0
    )
    
    return stratified_train, stratified_val, stratified_test

def create_stratified_split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create stratified split of data ensuring balanced label distribution
    """
    # Group data by labels
    label_groups = {}
    for doc, label in data:
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append((doc, label))
    
    train_data, val_data, test_data = [], [], []
    
    # Split each label group maintaining ratios
    for label in label_groups:
        items = label_groups[label]
        random.shuffle(items)
        n = len(items)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_data.extend(items[:train_end])
        val_data.extend(items[train_end:val_end])
        test_data.extend(items[val_end:])
    
    return train_data, val_data, test_data

def print_label_distribution(data, split_name):
    """Print label distribution in a data split"""
    labels = [y for _, y in data]
    dist = Counter(labels)
    total = len(labels)
    print(f"\n{split_name} label distribution:")
    for label in sorted(dist.keys()):
        print(f"Label {label}: {dist[label]} ({dist[label]/total*100:.1f}%)")
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    
    # Add new arguments
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--test_model", action='store_true', help="Test saved model")
    parser.add_argument("--output_dir", default="results", help="output directory for results")
    parser.add_argument("--use_stratified", action='store_true', help="Use stratified data split")
    parser.add_argument("--do_grid_search", action='store_true', help="Perform grid search over hyperparameters")
   
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data, val_data, test_data = load_data(args.train_data, args.val_data, "test.json") # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)
    # Analyze original distribution
    print("\nOriginal Data Distribution:")
    print_label_distribution(train_data, "Training")
    print_label_distribution(val_data, "Validation")
    print_label_distribution(test_data, "Test")
    
    if args.use_stratified:
        print("\nRebalancing data splits while maintaining original ratios...")
        # Only rebalance train and val sets
        train_data, val_data, _ = combine_and_stratify_data(train_data, val_data)
        print("\nStratified Data Distribution:")
        print_label_distribution(train_data, "Training")
        print_label_distribution(val_data, "Validation")
    # Track experiment type in metrics
    experiment_name = "stratified" if args.use_stratified else "original"

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    val_data = convert_to_vector_representation(val_data, word2index)
    # Add: vectorize test data
    test_data = convert_to_vector_representation(test_data, word2index)
    
    # Add: appropriate torch device
    if torch.backends.mps.is_available():
        device  = torch.device("mps")
        print("Using Apple MPS accelerated training")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using a CUDA device")
    else:
        device = model.device("cpu")
        print("No GPU found, using CPU")
        
    if args.do_grid_search:
        print("\nPerforming grid search over hyperparameters...")
        # use args.output_dir as dir structure handled within func
        grid_search_results = grid_search_hyperparameters(vocab, device, train_data, val_data, args.output_dir, args.use_stratified)
        
        # Print best results
        best_result = max(grid_search_results, key=lambda x: x['best_val_acc'])
        print("\nBest hyperparameters found:")
        print(f"Hidden dim: {best_result['hidden_dim']}")
        print(f"Dropout: {best_result['dropout']}")
        print(f"Learning rate: {best_result['learning_rate']}")
        print(f"Best validation accuracy: {best_result['best_val_acc']:.4f}")
        print(f"Best epoch: {best_result['best_epoch']}")
    else:
        if not args.test_model:
            # Add: dropout for regularization
            model = FFNN(input_dim=len(vocab), h=args.hidden_dim,  dropout_rate=args.dropout).to(device)
            optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
            print("========== Training for {} epochs ==========".format(args.epochs))
            
            # Track metrics across epochs
            epoch_metrics = []
            # Training loop
            for epoch in range(args.epochs):
                avg_train_loss, train_accuracy = training_loop(model, optimizer, device, train_data, epoch)
                # Validation loop
                avg_val_loss, val_accuracy = validation_loop(model, device, val_data, epoch)

                # Add: After validation, save metrics, analyze the errors
                epoch_metrics.append({
                    'epoch': epoch + 1,
                    'split_type': experiment_name,
                    'train_loss': avg_train_loss,
                    'train_acc': train_accuracy,
                    'val_loss': avg_val_loss,
                    'val_acc': val_accuracy
                })

                # Add: save metrics, analyze errors, create classification report
                # Save every epoch
                save_metrics(model, epoch_metrics, args.output_dir, args.use_stratified)
                # Analyze the final model
                if epoch == args.epochs - 1:
                    torch.save(model.state_dict(), f"{args.output_dir}/final_model.pt")
                    data_dict = {'val_report': val_data}
                    evaluation(model, data_dict, word2index, index2word)
        else:
            # Load model from saved state
            model = FFNN(input_dim=len(vocab), h=args.hidden_dim, dropout_rate=args.dropout).to(device)
            # Load to CPU first, then transfer to target device
            state_dict = torch.load(f"{args.output_dir}/best_model.pt", map_location='cpu')
            model.load_state_dict(state_dict)
            model = model.to(device)
            # model.load_state_dict(torch.load(f"{args.output_dir}/best_model.pt"), map_location=device)
            
            # Load data for evaluation
            data_dict = {'val_report': val_data, 'test_report': test_data}
            evaluation(model, data_dict, word2index, index2word, args.output_dir)
        

    # write out to results/test.out [DONE]
    