"""
    Beautiful resource for Optimizers, Schedulers: https://pytorch.org/docs/stable/optim.html
    Reference: https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html
    List of Optimizers tried: SGD, RMSprop, Adam, AdamW: optim.SGD, optim.RMSprop, optim.Adam, optim.AdamW
    List of Scheduler tried: lr_scheduler.ExponentialLR(optimizer, gamma=0.9), None

    wandb API key: 
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
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math # for math.sqrt

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
    parser.add_argument("--num_epochs", default=20, type=int, help="Number of Epochs")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Initial Learning Rate")
    parser.add_argument("--path_to_glove", default="./glove.6B.300d.txt", type=str, help="Path to pretrained GloVe embeddings")
    parser.add_argument("--path_to_corpus", default="./Auguste_Maquet.txt", type=str, help="Path to the corpus (.txt file)")
    # parser.add_argument("--test_size", default=20000, type=int, help="Test Size (number of datapoints)")
    # parser.add_argument("--valid_size", default=10000, type=int, help="Validation Size (number of datapoints)")
    parser.add_argument("--dropout_rate", default=0.3, type=float, help="Dropout rate for preventing overfitting")
    parser.add_argument("--optimizer", default="AdamW", type=str, help="Type of Optimizer")
    parser.add_argument("--scheduler", default=None, type=str, help="Scheduler for scheduling learning rate")
    parser.add_argument("--gamma", default=0.9, type=float, help="Gamma value for ExponentialLR scheduler")
    parser.add_argument("--use_wandb", default=False, type=bool, help="Enable wandb logging")
    parser.add_argument("--nhead", default=5, type=int, help="Number of heads in multi-head attention")
    parser.add_argument("--num_decoder_layers", default=6, type=int, help="Number of sub-decoder-layers in the decoder")
    parser.add_argument("--dim_feedforward", default=1028, type=int, help="Dimension of the feedforward network model (Typically 4 times embedding_dim in Transformer architecture)")
    parser.add_argument("--max_seq_len", default=64, type=int, help="Maximum sequence length of input to the Transformer")

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
    vocab['<UNK>'] = len(vocab) # Add unknown token
    vocab['<PAD>'] = len(vocab) # Add padding token
    idx2word = {idx: word for word, idx in vocab.items()}
    return vocab, idx2word

# Dataset
class Custom_Dataset(nn.Module):
    def __init__(self, input_sentences, target_sentences, vocab: dict, seq_length: int=64):
        self.input_sentences = input_sentences
        self.target_sentences = target_sentences
        self.seq_length = seq_length
        self.vocab = vocab

    def __len__(self):
        return len(self.input_sentences)
    
    def __getitem__(self, idx):
        input_idxs = [self.vocab.get(w, self.vocab['<UNK>']) for w in self.input_sentences[idx]]
        target_idxs = [self.vocab.get(w, self.vocab['<UNK>']) for w in self.target_sentences[idx]]

        return torch.tensor(input_idxs), torch.tensor(target_idxs)

#Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # t, t, t, ...
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # 10000*(2/d) 10000*(2/d) ...

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class TransformerDecoderLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_decoder_layers, dim_feedforward, dropout_rate=0.1):
        super(TransformerDecoderLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout_rate)
        decoder_layer = TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead, 
                                                dim_feedforward=dim_feedforward, 
                                                dropout=dropout_rate)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, inputs):
        # inputs shape: (batch_size, seq_len)
        embedded = self.embeddings(inputs) * math.sqrt(self.embeddings.embedding_dim)
        embedded = self.pos_encoder(embedded)
        
        # Create a square subsequent mask for the decoder
        mask = self.generate_square_subsequent_mask(inputs.size(1)).to(inputs.device)
        
        # Create a dummy memory input (all zeros) with the same batch size and embedding dimension (Necessary)
        memory = torch.zeros(inputs.size(1), inputs.size(0), self.embeddings.embedding_dim).to(inputs.device)
        
        # Pass through the transformer decoder
        # Note: TransformerDecoder expects input of shape (seq_len, batch_size, embedding_dim)
        output = self.transformer_decoder(embedded.transpose(0, 1), memory=memory, tgt_mask=mask)
        
        # Pass through the output layer
        output = self.output_layer(output.transpose(0, 1))
        return self.softmax(output)
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# Training function
def train(args, epoch: int, model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device, val_loader: DataLoader, scheduler: optim.lr_scheduler=None):
    model.train()
    total_loss = 0
    for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        optimizer.zero_grad()
        output = model(input_seq)
        loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if((batch_idx) % 40 == 0):
            print(f"Batch: {batch_idx}/{len(train_loader)}; Loss: {loss.item()}")
        

        # log to wandb
        if args.use_wandb == True:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/perplexity": torch.exp(loss).item(),
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
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (input_seq, target_seq) in enumerate(val_loader):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            output = model(input_seq)
            loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
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
def write_perplexity_to_file(args, input_sentences, target_sentences, model: nn.Module, vocab: dict, criterion: nn.Module, file_name, device):
    avg_perplexity = 0.0
    count_sents = 0
    model.eval()
    # Create a reverse vocabulary (index to word)
    idx2word = {idx: word for word, idx in vocab.items()}
    with open(file_name, "w") as file:
        dataset = Custom_Dataset(input_sentences, target_sentences, vocab=vocab, seq_length=args.max_seq_len)
        loader = DataLoader(dataset, batch_size=1)
        for batch_idx, (input_seq, target_seq) in enumerate(loader):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            output = model(input_seq)
            loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
            perplexity = torch.exp(loss).item()

            if not math.isnan(perplexity): # NaN whens only one word in a sentence
                avg_perplexity += perplexity
                count_sents += 1
            # Convert input_seq indices back to words, ignoring '<PAD>'
            input_words = [idx2word[idx.item()] for idx in input_seq[0] if idx2word[idx.item()] != '<PAD>']
            
            # Join the words and write to file
            file.write(f"{' '.join(input_words)}\t{perplexity:.4f}\n")

        avg_perplexity = avg_perplexity / count_sents
        file.write(f"Average Perplexity: {avg_perplexity:.4f}")
        
    

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

    random.shuffle(tokenized_corpus)

    total_size = len(tokenized_corpus)
    # valid_size = args.valid_size
    valid_size = int(0.1 * total_size)
    # test_size = args.test_size
    test_size = int(0.2 * total_size)
    train_size = total_size - (valid_size + test_size)

    pad_token = "<PAD>"
    input_tokenized_corpus = []
    target_tokenized_corpus = []

    for sentence in tokenized_corpus:
        for i in range(0, len(sentence), args.max_seq_len - 1):
            input_seq = sentence[i: i + args.max_seq_len - 1]

            if len(input_seq) < args.max_seq_len:
                input_seq += [pad_token] * (args.max_seq_len - len(input_seq))

            target_seq = input_seq[1:] + [pad_token]

            input_tokenized_corpus.append(input_seq)
            target_tokenized_corpus.append(target_seq)

    print("Input Sequences: ", input_tokenized_corpus[:1])
    print("len: ", len(input_tokenized_corpus[0]))
    print("Target Sequences: ", target_tokenized_corpus[:1])
    print("len: ", len(target_tokenized_corpus[0]))

    # split the sentences into train, test and validation
    input_train_sentences = input_tokenized_corpus[:train_size]
    input_valid_sentences = input_tokenized_corpus[train_size: train_size + valid_size]
    input_test_sentences = input_tokenized_corpus[train_size + valid_size:]
    target_train_sentences = target_tokenized_corpus[:train_size]
    target_valid_sentences = target_tokenized_corpus[train_size: train_size + valid_size]
    target_test_sentences = target_tokenized_corpus[train_size + valid_size:]

    print("total_sentences: ", total_size)
    print("train_sentences: ", len(input_train_sentences))
    print("valid_sentences: ", len(input_valid_sentences))
    print("test_sentences: ", len(input_test_sentences))
    print("train_sentences: ", len(target_train_sentences))
    print("valid_sentences: ", len(target_valid_sentences))
    print("test_sentences: ", len(target_test_sentences))

    # sys.exit()

    train_tokenized_corpus = tokenized_corpus[:train_size]
    tokens = []
    for sentence in train_tokenized_corpus: # Only training corpus vocab should be there in vocab ***
        for token in sentence:
            tokens.append(token)
    
    print("Text file preprocessed")
    vocab, idx2word = build_vocab(tokens)
    print("vocabulary created for complete corpus")
    
    # Create dataset and dataloader
    train_dataset = Custom_Dataset(input_train_sentences, target_train_sentences, vocab, seq_length=args.max_seq_len)
    valid_dataset = Custom_Dataset(input_valid_sentences, target_valid_sentences, vocab, seq_length=args.max_seq_len)
    test_dataset = Custom_Dataset(input_test_sentences, target_test_sentences, vocab, seq_length=args.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    print("Data Loaded Successfully")

    # sys.exit()

    model_path = f"models/model.pth"
    
    # Initialize model, optimizer, and loss function
    model = TransformerDecoderLM(len(vocab), args.embedding_dim, args.nhead, args.num_decoder_layers, args.dim_feedforward, args.dropout_rate).to(device)
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
            'configs': args,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab': vocab,
            'input_train_sentences': input_train_sentences,
            'input_test_sentences': input_test_sentences,
            'input_valid_sentences': input_valid_sentences,
            'target_train_sentences': target_train_sentences,
            'target_test_sentences': target_test_sentences,
            'target_valid_sentences': target_valid_sentences,
        }, model_save_path)
        print(f"Model and training state saved to {model_save_path}")

    # Write train and test sentences with perplexity scores to respective files
    write_perplexity_to_file(args, input_train_sentences, target_train_sentences, model, vocab, criterion, "./models/train.txt", device)
    write_perplexity_to_file(args, input_test_sentences, target_test_sentences, model, vocab, criterion, "./models/test.txt", device)
    print("Perplexity scores written to train.txt and test.txt")

    if args.use_wandb == True:
        wandb.finish()

if __name__ == '__main__':
    main()
