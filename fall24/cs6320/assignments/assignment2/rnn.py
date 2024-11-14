import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

# Add imports
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Import utilities from ffnn
from ffnn import (load_data, save_metrics,
                 print_label_distribution, combine_and_stratify_data)

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h, numOfLayers, dropout):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayers = numOfLayers
        self.rnn = nn.RNN(input_dim, h, self.numOfLayers, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.dropout = nn.Dropout(dropout)  # Add dropout
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # [to fill] obtain hidden layer representation
        _, hidden = self.rnn(inputs)
        # [to fill] obtain output layer representations
        output = self.W(self.dropout(hidden[-1]))
        # [to fill] sum over output
        final_output = output
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(final_output)
        
        return predicted_vector

def training_loop(model, optimizer, device, train_data, epoch, word_embedding, batch_size=16):
    """Single training epoch for RNN"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    
    print(f"Training started for epoch {epoch + 1}")
    random.shuffle(train_data)
    
    N = len(train_data)
    num_batches = 0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    for minibatch_index in tqdm(range(N // batch_size)):
        optimizer.zero_grad()
        batch_loss = None
        
        for example_index in range(batch_size):
            idx = minibatch_index * batch_size + example_index
            if idx >= N:
                break
                
            # Process input
            input_words, gold_label = train_data[idx]
            input_words = " ".join(input_words)
            input_words = input_words.translate(
                input_words.maketrans("", "", string.punctuation)
            ).split()
            
            # Convert words to indices
            indices = torch.tensor(np.array([
                word_embedding.get(w.lower(), word_embedding['unk'])
                for w in input_words
            ])).to(device)
            
            predicted_vector = model(indices.unsqueeze(1))
            predicted_label = torch.argmax(predicted_vector)
            
            correct += int(predicted_label == gold_label)
            total += 1
            
            example_loss = model.compute_Loss(
                predicted_vector.view(1,-1), 
                torch.tensor([gold_label]).to(device)
            )
            
            if batch_loss is None:
                batch_loss = example_loss
            else:
                batch_loss += example_loss
        
        if batch_loss is not None:
            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    accuracy = correct / total
    
    print(f"Training completed for epoch {epoch + 1}")
    print(f"Training accuracy: {accuracy}")
    print(f"Training loss: {avg_loss}")
    print(f"Training time: {time.time() - start_time}")
    
    return avg_loss, accuracy

def validation_loop(model, device, val_data, epoch, word_embedding, batch_size=16):
    """Single validation epoch for RNN"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    
    print(f"Validation started for epoch {epoch + 1}")
    N = len(val_data)
    num_batches = 0
    
    with torch.no_grad():
        for minibatch_index in tqdm(range(N // batch_size)):
            batch_loss = None
            
            for example_index in range(batch_size):
                idx = minibatch_index * batch_size + example_index
                if idx >= N:
                    break
                
                # Process input
                input_words, gold_label = val_data[idx]
                input_words = " ".join(input_words)
                input_words = input_words.translate(
                    input_words.maketrans("", "", string.punctuation)
                ).split()
                
                # Convert words to indices
                indices = torch.tensor(np.array([
                    word_embedding.get(w.lower(), word_embedding['unk'])
                    for w in input_words
                ])).to(device)
                
                predicted_vector = model(indices.unsqueeze(1))
                predicted_label = torch.argmax(predicted_vector)
                
                correct += int(predicted_label == gold_label)
                total += 1
                
                example_loss = model.compute_Loss(
                    predicted_vector.view(1,-1), 
                    torch.tensor([gold_label]).to(device)
                )
                
                if batch_loss is None:
                    batch_loss = example_loss
                else:
                    batch_loss += example_loss
            
            if batch_loss is not None:
                batch_loss = batch_loss / batch_size
                total_loss += batch_loss.item()
                num_batches += 1
    
    avg_loss = total_loss / num_batches
    accuracy = correct / total
    
    print(f"Validation completed for epoch {epoch + 1}")
    print(f"Validation accuracy: {accuracy}")
    print(f"Validation loss: {avg_loss}")
    print(f"Validation time: {time.time() - start_time}")
    
    return avg_loss, accuracy

def grid_search_hyperparameters_rnn(train_data, val_data, base_output_dir, word_embedding, device, use_stratified):
    """
    Perform grid search over hyperparameters for RNN
    Returns: List of dicts containing results for each configuration
    """
    # Define parameter grid
    param_grid = {
        'hidden_dim': [64, 128],
        'layers': [1, 2, 3],  # Start shallow
        'dropout': [0.2],
        'learning_rate': [0.001]  # Lower learning rates
    }
    
    results = []
    
    # Generate all combinations
    for hidden_dim in param_grid['hidden_dim']:
        for layers in param_grid['layers']:
            for dropout in param_grid['dropout']:
                for lr in param_grid['learning_rate']:
                    # Create unique experiment name
                    exp_name = f"hd{hidden_dim}_l{layers}_dr{dropout}_lr{lr}"
                    dataset_name = "stratified" if use_stratified else "original"
                    exp_dir = os.path.join(base_output_dir, exp_name)
                    os.makedirs(exp_dir, exist_ok=True)
                    
                    print(f"\nStarting experiment: {exp_name}")
                    print("=" * 50)
                    print(f"Hidden dim: {hidden_dim}")
                    print(f"Layers: {layers}")
                    print(f"Dropout: {dropout}")
                    print(f"Learning rate: {lr}")
                    
                    # Initialize model and optimizer
                    model = RNN(
                        input_dim=50,  # embedding dimension
                        h=hidden_dim,
                        numOfLayers=layers,
                        dropout=dropout
                    ).to(device)
                    
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    
                    # Training loop with early stopping
                    best_val_acc = 0
                    best_epoch = 0
                    patience = 5
                    no_improve_count = 0
                    epoch_metrics = []
                    
                    for epoch in range(20):  # Max epochs
                        train_loss, train_acc = training_loop(
                            model, optimizer, device, train_data, epoch, 
                            word_embedding
                        )
                        
                        val_loss, val_acc = validation_loop(
                            model, device, val_data, epoch,
                            word_embedding
                        )
                        
                        # Save epoch metrics
                        metrics = {
                            'epoch': epoch + 1,
                            'train_loss': train_loss,
                            'train_acc': train_acc,
                            'val_loss': val_loss,
                            'val_acc': val_acc,
                            'hidden_dim': hidden_dim,
                            'layers': layers,
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
                        'layers': layers,
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

def analyze_errors_rnn(model, data_loader, word_embedding, output_dir):
    """Analyze RNN model errors and generate confusion matrix"""
    all_preds = []
    all_labels = []
    error_examples = []
    
    model.eval()
    with torch.no_grad():
        for input_words, gold_label in tqdm(data_loader, desc="Analyzing errors"):
            # Process input words
            input_text = " ".join(input_words)
            clean_words = input_text.translate(
                str.maketrans("", "", string.punctuation)
            ).split()
            
            # Convert to embeddings
            indices = torch.tensor([
                word_embedding.get(w.lower(), word_embedding['unk'])
                for w in clean_words
            ]).unsqueeze(1).to(device)  # Add batch dimension
            
            pred = model(indices)
            pred_label = torch.argmax(pred).item()
            
            if pred_label != gold_label:
                error_examples.append({
                    'text': input_text,
                    'predicted': pred_label,
                    'actual': gold_label,
                    'length': len(clean_words)  # Track sequence length
                })
            
            all_preds.append(pred_label)
            all_labels.append(gold_label)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('RNN Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()
    
    # Additional RNN-specific analysis
    # Analyze error distribution by sequence length
    error_df = pd.DataFrame(error_examples)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=error_df, x='length', bins=30)
    plt.title('Error Distribution by Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.savefig(f"{output_dir}/error_by_length.png")
    plt.close()
    
    # Save error examples
    error_df.to_csv(f"{output_dir}/error_examples.csv", index=False)
    
    # Save summary statistics
    with open(f"{output_dir}/error_analysis.txt", "w") as f:
        f.write("Error Analysis Summary\n")
        f.write(f"Total samples: {len(all_labels)}\n")
        f.write(f"Error rate: {len(error_examples)/len(all_labels):.2%}\n")
        f.write(f"\nSequence Length Statistics:\n")
        f.write(f"Mean: {error_df['length'].mean():.2f}\n")
        f.write(f"Median: {error_df['length'].median():.2f}\n")
        f.write(f"Max: {error_df['length'].max()}\n")
        f.write(f"Min: {error_df['length'].min()}\n")

def evaluation_rnn(model, data_dict, word_embedding, output_dir):
    """Evaluate RNN model performance"""
    # First analyze errors if test set is available
    if 'test_report' in data_dict:
        analyze_errors_rnn(model, data_dict['test_report'], word_embedding, output_dir)
    
    # Generate classification reports
    for report_name in data_dict.keys():
        predicted_labels = []
        true_labels = []
        model.eval()
        
        with torch.no_grad():
            for input_words, gold_label in data_dict[report_name]:
                # Process input
                clean_words = " ".join(input_words).translate(
                    str.maketrans("", "", string.punctuation)
                ).split()
                
                # Convert to embeddings
                indices = torch.tensor([
                    word_embedding.get(w.lower(), word_embedding['unk'])
                    for w in clean_words
                ]).unsqueeze(1).to(device)
                
                pred = model(indices)
                predicted_labels.append(torch.argmax(pred).item())
                true_labels.append(gold_label)
        
        # Generate and save classification report
        report = classification_report(true_labels, predicted_labels)
        with open(f"{output_dir}/{report_name}.txt", "w") as f:
            f.write(report)
        
        # Additional RNN-specific metrics
        with open(f"{output_dir}/{report_name}_analysis.txt", "w") as f:
            f.write("\nSequence Length Analysis:\n")
            lengths = [len(words) for words, _ in data_dict[report_name]]
            f.write(f"Mean sequence length: {np.mean(lengths):.2f}\n")
            f.write(f"Median sequence length: {np.median(lengths):.2f}\n")
            f.write(f"Max sequence length: {max(lengths)}\n")
            f.write(f"Min sequence length: {min(lengths)}\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    
    # Add new arguments
    parser.add_argument("--layers", type=int, default=1, help="no of layers in the RNN")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--output_dir", default="results", help="output directory")
    parser.add_argument("--use_stratified", action='store_true', help="use stratified data split")
    parser.add_argument("--do_grid_search", action='store_true', help="perform hyperparameter search")
    parser.add_argument("--test_model", action='store_true', help="Test saved model")
    # parser.add_argument("--fine_tune_embeddings", action='store_true', help="fine-tune word embeddings during training")
    args = parser.parse_args()

    print("========== Loading data ==========")
    os.makedirs(args.output_dir, exist_ok=True)
    train_data, val_data, test_data = load_data(args.train_data, args.val_data, "test.json") # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    if args.use_stratified:
        print("\nRebalancing data splits while maintaining original ratios...")
        train_data, val_data, _ = combine_and_stratify_data(train_data, val_data)

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

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

    word_embedding = pickle.load(open('./Data_Embedding/word_embedding.pkl', 'rb'))

    if args.do_grid_search:
        print("\nPerforming grid search over hyperparameters...")
        results = grid_search_hyperparameters_rnn(
            train_data, val_data, word_embedding, device,
            args.output_dir, args.use_stratified
        )
        
        best_result = max(results, key=lambda x: x['best_val_acc'])
        print("\nBest hyperparameters found:")
        print(f"Hidden dim: {best_result['hidden_dim']}")
        print(f"Layers: {best_result['layers']}")
        print(f"Dropout: {best_result['dropout']}")
        print(f"Learning rate: {best_result['learning_rate']}")
        print(f"Best validation accuracy: {best_result['best_val_acc']:.4f}")
        print(f"Best epoch: {best_result['best_epoch']}")
    else:
        if not args.test_model:
            model = RNN(
                input_dim=50,
                h=args.hidden_dim,
                numOfLayers=args.layers,
                dropout=args.dropout
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            
            epoch_metrics = []
            for epoch in range(args.epochs):
                train_loss, train_acc = training_loop(
                    model, optimizer, device, train_data, 
                    epoch, word_embedding
                )
                val_loss, val_acc = validation_loop(
                    model, device, val_data, epoch, word_embedding
                )
                
                metrics = {
                    'epoch': epoch + 1,
                    'split_type': experiment_name,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }
                epoch_metrics.append(metrics)
                save_metrics(model, epoch_metrics, args.output_dir)
                
                if epoch == args.epochs - 1:
                    torch.save(model.state_dict(), 
                            f"{args.output_dir}/final_model.pt")
                    data_dict = {'val_report': val_data}
                    evaluation_rnn(model, data_dict, word_embedding)
        else:
            model = RNN(
                input_dim=50,
                h=args.hidden_dim,
                numOfLayers=args.layers,
                dropout=args.dropout
            ).to(device)
            state_dict = torch.load(f"{args.output_dir}/best_model.pt", 
                                map_location='cpu')
            model.load_state_dict(state_dict)
            model = model.to(device)
            
            data_dict = {'val_report': val_data, 'test_report': test_data}
            evaluation_rnn(model, data_dict, word_embedding, args.output_dir)


    # You may find it beneficial to keep track of training accuracy or training loss;

    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance
