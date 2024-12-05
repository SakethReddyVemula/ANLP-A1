"""
    Beautiful resource for Optimizers, Schedulers: https://pytorch.org/docs/stable/optim.html
    List of Optimizers tried: SGD, RMSprop, Adam, AdamW: optim.SGD, optim.RMSprop, optim.Adam, optim.AdamW
    List of Scheduler tried: lr_scheduler.ExponentialLR(optimizer, gamma=0.9), None

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
from typing import List, Tuple
# from torch.nn import DataParallel
import argparse
import random
import sys
import math

# Set environment for wandb
import wandb
import os

# Parse arguments
# EMBEDDING_DIM = 300
# HIDDEN_DIM = 256
# BATCH_SIZE = 64
# EPOCHS = 10
# LEARNING_RATE = 0.001
# GLOVE_FILE = "GloVe/glove.6B/glove.6B.300d.txt"
# CORPUS_PATH = "Auguste_Maquet.txt"
# TEST_SIZE = 20000
# VALID_SIZE = 10000


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_dim", default=300, type=int, help="Embedding Dimension")
    parser.add_argument("--hidden_dim", default=128, type=int, help="Dimension of each Hidden Layer")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--num_epochs", default=10, type=int, help="Number of Epochs")
    parser.add_argument("--learning_rate", default=5e-4, type=float, help="Initial Learning Rate")
    parser.add_argument("--path_to_glove", default="./glove.6B.300d.txt", type=str, help="Path to pretrained GloVe embeddings")
    parser.add_argument("--path_to_corpus", default="./Auguste_Maquet.txt", type=str, help="Path to the corpus (.txt file)")
    parser.add_argument("--dropout_rate", default=0.3, type=float, help="Dropout rate for preventing overfitting")
    parser.add_argument("--optimizer", default="AdamW", type=str, help="Type of Optimizer")
    parser.add_argument("--scheduler", default=None, type=str, help="Scheduler for scheduling learning rate")
    parser.add_argument("--gamma", default=0.9, type=float, help="Gamma value for ExponentialLR scheduler")
    parser.add_argument("--use_wandb", default=False, type=bool, help="Enable wandb logging")


    args = parser.parse_args()
    return args


# Data Preprocessing
class Tokenizer:
    def __init__(self):
        pass

    def tokenize_corpus(self, text):
        # Sentence Tokenizer
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        tokenized_text = []

        for sentence in sentences:
            tokenized_sentence = self.tokenize_sentence(sentence)
            tokenized_text.append(tokenized_sentence)

        return tokenized_text
    
    def tokenize_sentence(self, sentence):
        # Replace "\" with space
        sentence = re.sub(r'\\', r' ', sentence)

        # Time
        sentence = re.sub(r'\d:\d{2} [AP]M', r'', sentence)

        # Mentions
        sentence = re.sub(r'@[_\w]+', r'', sentence)

        # Hashtags
        sentence = re.sub(r'#[_\w]+', r'', sentence)

        # Mail IDs
        sentence = re.sub(r'[\w!%+-\.\/]+@[a-zA-Z0-9\.]*[a-zA-Z0-9]+|".+"@[a-zA-Z0-9\.]*[a-zA-Z0-9]+', r'', sentence)

        # URLs
        sentence = re.sub(r'(?:[a-z][a-z0-9+.-]*):\/\/(?:(?:[a-z]+)(?::(?:[^@]+))?@)?(?:[a-zA-Z0-9\.-]+|\[[a-fA-F0-9:]+\])(?::(?:\d+))?\/?(?:[\/a-z-A-Z0-9\_]*)?(?:\?(?:[^#]*))?(?:#(?:.*))?|(?:(?:\w+\.)+(?:\w+))(?::(?:\d+))?\/?(?:[\/a-z-A-Z0-9\_]*)?(?:\?(?:[^#]*))?(?:#(?:.*))?', r'', sentence)

        # Numbers
        sentence = re.sub(r'\d+.\d+|\d+|-\d+|\+\d+|\.\d+', r'', sentence)

        # # Punctuation
        # sentence = re.sub(r'^[^\w\s\<\>]$', r'', sentence)
        # Punctuation
        sentence = re.sub(r'[^\w\s]', r'', sentence)

        # Mobile Numbers
        sentence = re.sub(r'[\+0-9\-\(\)\.]{3,}[\-\.]?[0-9\-\.]{3,}', r'', sentence)

        # lower case
        sentence = sentence.lower()

        # return sentence
        return sentence.split()

def load_glove_embeddings(glove_file: str, vocab: dict, embedding_dim: int) -> torch.Tensor:
    embeddings = torch.zeros(len(vocab), embedding_dim)
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32)
            if word in vocab:
                embeddings[vocab[word]] = vector
    return embeddings


def build_vocab(tokens: List[str]) -> Tuple[dict, dict]:
    word_counts = Counter(tokens)
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
    vocab['<UNK>'] = len(vocab)
    idx2word = {idx: word for word, idx in vocab.items()}
    return vocab, idx2word

# Dataset
class NGramDataset(Dataset):
    def __init__(self, tokens: List[str], vocab: dict, context_size: int = 5):
        self.tokens = tokens
        self.vocab = vocab
        self.context_size = context_size
        
    def __len__(self):
        return len(self.tokens) - self.context_size
    
    def __getitem__(self, idx):
        context = self.tokens[idx:idx+self.context_size]
        target = self.tokens[idx+self.context_size]
        context_ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in context]
        target_id = self.vocab.get(target, self.vocab['<UNK>'])
        return torch.tensor(context_ids), torch.tensor(target_id)

# Model
class NeuralLM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout_rate: float = 0.1):
        super(NeuralLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden1 = nn.Linear(embedding_dim * 5, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate) # Regularisation technique that helps in preventing overfitting (critical) # Automatically set to 0.0 in model.eval() mode.
        self.hidden2 = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((-1, 5 * self.embeddings.embedding_dim))
        hidden1_out = torch.relu(self.hidden1(embeds)) # Activation function
        hidden1_out = self.dropout(hidden1_out) # applying dropout after first hidden layer
        hidden2_out = self.hidden2(hidden1_out)
        return self.softmax(hidden2_out)

# Training function
def train(args, epoch: int, model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device, val_loader: DataLoader, scheduler: optim.lr_scheduler=None):
    model.train()
    total_loss = 0
    for batch_idx, (context, target) in enumerate(train_loader):
        context, target = context.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(context)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if((batch_idx) % 1000 == 0):
            print(f"Batch: {batch_idx}/{len(train_loader)}; Loss: {loss.item()}")
        
        # log to wandb
        if args.use_wandb == True:  
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/perplexxity": torch.exp(loss).item(),
                    "stats/learning_rate": optimizer.param_groups[0]['lr']
                },
                step = batch_idx + epoch * len(train_loader) + epoch * len(val_loader)# Effectively global step
            )
    
    if scheduler:
        scheduler.step()

    avg_loss = total_loss / len(train_loader)
    avg_perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, avg_perplexity.item()

# Evaluation function
@torch.no_grad()
def evaluate(args, epoch: int, model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device, train_loader: DataLoader):
    model.eval() # Sets dropout to 0
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (context, target) in enumerate(val_loader):
            context, target = context.to(device), target.to(device)
            output = model(context)
            loss = criterion(output, target)
            total_loss += loss.item()

            if args.use_wandb == True:
                wandb.log(
                    { 
                        'valid/loss': loss.item(),
                        'valid/perplexity': torch.exp(loss).item()
                    },
                    step = batch_idx + (epoch + 1) * len(train_loader) + epoch * len(val_loader)
                )


    avg_loss = total_loss / len(val_loader)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity.item()

# Calculate and write perplexity scores to files
@torch.no_grad()
def write_perplexity_to_file(args, sentences, model: nn.Module, vocab: dict, criterion: nn.Module, file_name, device):
    avg_perplexity = 0.0
    count_sents = 0
    with open(file_name, "w") as file:
        for sentence in sentences:
            if len(sentence) < 6:
                continue
            
            tokens = []
            for token in sentence:
                if token != '<PAD>': # Otherwise giving NaNs
                    tokens.append(token)
            
            if len(tokens) <= 5:
                continue

            dataset = NGramDataset(tokens, vocab)
            loader = DataLoader(dataset, batch_size=1)

            if len(loader) == 0:
                continue

            count_sents += 1
            sent_loss = 0.0
            for batch_idx, (context, target) in enumerate(loader):
                context, target = context.to(device), target.to(device)
                output = model(context)
                loss = criterion(output, target)
                if not math.isnan(loss.item()):
                    sent_loss += loss.item()

            sent_loss = sent_loss / len(loader)

            sent_perplexity = torch.exp(torch.tensor(sent_loss)).item()
            avg_perplexity += sent_perplexity
            write_sentence = []
            for word in sentence:
                if word != '<PAD>':
                    write_sentence.append(word)
            file.write(f"{' '.join(write_sentence)}\t{sent_perplexity:.4f}\n")

        avg_perplexity = avg_perplexity / count_sents

        file.write(f"Average Perplexity: {avg_perplexity}\n")
    

# Main function
def main():
    # Hyperparameters
    # EMBEDDING_DIM = 300
    # HIDDEN_DIM = 256
    # BATCH_SIZE = 64
    # EPOCHS = 10
    # LEARNING_RATE = 0.001
    # GLOVE_FILE = "GloVe/glove.6B/glove.6B.300d.txt"
    # CORPUS_PATH = "Auguste_Maquet.txt"
    # TEST_SIZE = 20000
    # VALID_SIZE = 10000

    args = parse_arguments() # Parse Arguments

    # Initialize WandB
    if args.use_wandb == True:
        os.environ["WANDB_API_KEY"] = ""
        args.wandb_id = wandb.util.generate_id()
        wandb.init(
            name=f"emb_{args.embedding_dim}_hd_{args.hidden_dim}_bs_{args.batch_size}_NNLM",
            config=args,
            id=args.wandb_id,
            project="Language_Models",
            entity="vemulasakethreddy_10"
        )
        print("wandb initialized successfully!")
    else:
        os.environ["WANDB_MODE"] = "disabled"

    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")
    
    # Load and preprocess data
    with open(args.path_to_corpus, "r") as file:
        raw_text = file.read()

    tokenizer = Tokenizer()
    tokenized_corpus = tokenizer.tokenize_corpus(raw_text)
    padded_tokenized_corpus = []
    for sentence in tokenized_corpus:
        padded_tokenized_corpus.append(['<PAD>'] * 5 + sentence + ['<PAD>'] * 5)

    random.shuffle(padded_tokenized_corpus)

    total_size = len(padded_tokenized_corpus)
    # valid_size = args.valid_size
    valid_size = int(0.1 * total_size)
    # test_size = args.test_size
    test_size = int(0.2 * total_size)
    train_size = total_size - (valid_size + test_size)

    
    # split the sentences into train, test and validation
    train_sentences = padded_tokenized_corpus[:train_size]
    valid_sentences = padded_tokenized_corpus[train_size: train_size + valid_size]
    test_sentences = padded_tokenized_corpus[train_size + valid_size:]

    print("total_sentences: ", total_size)
    print("train_sentences: ", len(train_sentences))
    print("valid_sentences: ", len(valid_sentences))
    print("test_sentences: ", len(test_sentences))

    print(padded_tokenized_corpus[:3])

    # sys.exit()

    tokens = []
    for sentence in padded_tokenized_corpus:
        for token in sentence:
            tokens.append(token)

    train_tokens = []
    for sentence in train_sentences:
        for token in sentence:
            train_tokens.append(token)

    valid_tokens = []
    for sentence in valid_sentences:
        for token in sentence:
            valid_tokens.append(token)
    
    test_tokens = []
    for sentence in test_sentences:
        for token in sentence:
            test_tokens.append(token)
    
    
    print("Text file preprocessed")
    vocab, idx2word = build_vocab(train_tokens)
    print("vocabulary created for complete corpus")
    
    # Create dataset and dataloader
    train_dataset = NGramDataset(train_tokens, vocab)
    valid_dataset = NGramDataset(valid_tokens, vocab)
    test_dataset = NGramDataset(test_tokens, vocab)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    print("Data Loaded Successfully")

    model_path = f"models/model.pth"
    
    # Initialize model, optimizer, and loss function
    model = NeuralLM(len(vocab), args.embedding_dim, args.hidden_dim).to(device)
    # if torch.cuda.device_count() > 1: # Use multiple GPUs. Data will be split automatically by torch.nn.DataParallel. No need to do manually later.
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model)
    model = model.to(device)

    if args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "RMSprop":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unsupported optimzer: {args.optimizer}")
    
    # Initialize the scheduler if specified
    if args.scheduler == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    else:
        scheduler = None

    criterion = nn.NLLLoss(ignore_index=vocab['<PAD>'])

    if os.path.exists(model_path):
        print("Model found. Skipping training and testing")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # vocab = checkpoint['vocab']
        # embedding_dim = checkpoint['embedding_dim']
        # hidden_dim = checkpoint['hidden_dim']
        model.eval()
    else:
        # Load pre-trained embeddings
        pretrained_embeddings = load_glove_embeddings(args.path_to_glove, vocab, args.embedding_dim)
        model.embeddings.weight.data.copy_(pretrained_embeddings)
        print("Loaded pretrained GloVe Embeddings")
        
        # Training loop
        overall_train_perplexity = 0
        overall_val_perplexity = 0
        for epoch in range(args.num_epochs):
            print(f"Training for Epoch: {epoch + 1}/{args.num_epochs}")
            avg_train_loss, avg_train_perplexity = train(args, epoch, model, train_loader, optimizer, criterion, device, valid_loader, scheduler)
            avg_val_loss, avg_val_perplexity = evaluate(args, epoch, model, valid_loader, criterion, device, train_loader)
            # print("-----------------------------------------------------------------------------------------------------\n")
            print(f'Epoch {epoch+1}/{args.num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Perplexity: {avg_train_perplexity:.4f}, Val Perplexity: {avg_val_perplexity:.4f}')
            print("-----------------------------------------------------------------------------------------------------")
            
            if args.use_wandb == True:
                wandb.log({
                    "epoch": epoch,
                    "train/avg_loss": avg_train_loss,
                    "train/avg_perplexity": avg_train_perplexity,
                    "valid/avg_loss": avg_val_loss,
                    "valid/avg_perplexity": avg_val_perplexity
                })
            
            overall_train_perplexity += avg_train_perplexity
            overall_val_perplexity += avg_val_perplexity

        overall_train_perplexity /= args.num_epochs
        overall_val_perplexity /= args.num_epochs
        print(f"Overall Train Perplexity: {overall_train_perplexity}")
        print(f"Overall Validation Perplexity: {overall_val_perplexity}")


        print(f"Training done...\n")
        test_loss, test_perplexity = evaluate(args, args.num_epochs, model, test_loader, criterion, device, train_loader)
        print(f"Final Test Loss: {test_loss:.4f}, Final Test Perplexity: {test_perplexity:.4f}")

        # Save model state, optimizer state, and epoch
        model_save_path = model_path
        torch.save({
            'epoch': args.num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab': vocab,
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'train_sentences': train_sentences,
            'test_sentences': test_sentences,
            'valid_sentences': valid_sentences
        }, model_save_path)
        print(f"Model and training state saved to {model_save_path}")

    # Write train and test sentences with perplexity scores to respective files
    write_perplexity_to_file(args, train_sentences, model, vocab, criterion, "./models/train.txt", device)
    write_perplexity_to_file(args, test_sentences, model, vocab, criterion, "./models/test.txt", device)
    print("Perplexity scores written to train.txt and test.txt")

    if args.use_wandb == True:
        wandb.finish()

if __name__ == '__main__':
    main()
