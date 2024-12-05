# A1-Language Modelling (ANLP)

Saketh Reddy Vemula
2022114014
CLD UG3

---
## Note:
`LM1_x`: feed forward models with different hyperparameters

`LM2_2`: Language model with multiple layers of LSTMs

`LM3_1`: Decoder-only Transformer model

## File Structure

```
.
├── 2022114014-LM1_1-test-perplexity.txt
├── 2022114014-LM1_1-train-perplexity.txt
├── 2022114014-LM1_2-test-perplexity.txt
├── 2022114014-LM1_2-train_perplexity.txt
├── 2022114014-LM1_3-test-perplexity.txt
├── 2022114014-LM1_3-train-perplexity.txt
├── 2022114014-LM1_4-test-perplexity.txt
├── 2022114014-LM1_4-train-perplexity.txt
├── 2022114014-LM1_5-test-perplexity.txt
├── 2022114014-LM1_5-train-perplexity.txt
├── 2022114014-LM2_2-test-perplexity.txt
├── 2022114014-LM2_2-train-perplexity.txt
├── 2022114014-LM3_1-test-perplexity.txt
├── 2022114014-LM3_1-train-perplexity.txt
├── models
│   ├── LM1_1
│   │   └── models
│   │       └── model.pth
│   ├── LM1_2
│   │   └── models
│   │       └── model.pth
│   ├── LM1_3
│   │   └── models
│   │       └── model.pth
│   ├── LM1_4
│   │   └── models
│   │       └── model.pth
│   ├── LM1_5
│   │   └── models
│   │       └── model.pth
│   ├── LM2_2
│   │   └── models
│   │       └── model.pth
│   └── LM3_1
│       └── models
│           └── model.pth
├── nnlm.py
├── README.md
├── Report.pdf
├── rnn.py
└── transformers.py

15 directories, 26 files

```

## Directions for Executing the codes

1. `nnlm.py`:
```
usage: nnlm.py [-h] [--embedding_dim EMBEDDING_DIM] [--hidden_dim HIDDEN_DIM] [--batch_size BATCH_SIZE]
               [--num_epochs NUM_EPOCHS] [--learning_rate LEARNING_RATE] [--path_to_glove PATH_TO_GLOVE]
               [--path_to_corpus PATH_TO_CORPUS] [--dropout_rate DROPOUT_RATE] [--optimizer OPTIMIZER]
               [--scheduler SCHEDULER] [--gamma GAMMA] [--use_wandb USE_WANDB]

options:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Embedding Dimension
  --hidden_dim HIDDEN_DIM
                        Dimension of each Hidden Layer
  --batch_size BATCH_SIZE
                        Batch size
  --num_epochs NUM_EPOCHS
                        Number of Epochs
  --learning_rate LEARNING_RATE
                        Initial Learning Rate
  --path_to_glove PATH_TO_GLOVE
                        Path to pretrained GloVe embeddings
  --path_to_corpus PATH_TO_CORPUS
                        Path to the corpus (.txt file)
  --dropout_rate DROPOUT_RATE
                        Dropout rate for preventing overfitting
  --optimizer OPTIMIZER
                        Type of Optimizer
  --scheduler SCHEDULER
                        Scheduler for scheduling learning rate
  --gamma GAMMA         Gamma value for ExponentialLR scheduler
  --use_wandb USE_WANDB
                        Enable wandb logging
```

2. `rnn.py`:
```
usage: rnn.py [-h] [--embedding_dim EMBEDDING_DIM] [--hidden_dim HIDDEN_DIM] [--batch_size BATCH_SIZE]
              [--num_epochs NUM_EPOCHS] [--learning_rate LEARNING_RATE] [--path_to_glove PATH_TO_GLOVE]
              [--path_to_corpus PATH_TO_CORPUS] [--dropout_rate DROPOUT_RATE] [--optimizer OPTIMIZER]
              [--scheduler SCHEDULER] [--gamma GAMMA] [--use_wandb USE_WANDB] [--num_layers NUM_LAYERS]
              [--max_seq_len MAX_SEQ_LEN]

options:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Embedding Dimension
  --hidden_dim HIDDEN_DIM
                        Dimension of each Hidden Layer
  --batch_size BATCH_SIZE
                        Batch size
  --num_epochs NUM_EPOCHS
                        Number of Epochs
  --learning_rate LEARNING_RATE
                        Initial Learning Rate
  --path_to_glove PATH_TO_GLOVE
                        Path to pretrained GloVe embeddings
  --path_to_corpus PATH_TO_CORPUS
                        Path to the corpus (.txt file)
  --dropout_rate DROPOUT_RATE
                        Dropout rate for preventing overfitting
  --optimizer OPTIMIZER
                        Type of Optimizer
  --scheduler SCHEDULER
                        Scheduler for scheduling learning rate
  --gamma GAMMA         Gamma value for ExponentialLR scheduler
  --use_wandb USE_WANDB
                        Enable wandb logging
  --num_layers NUM_LAYERS
                        Number of repetitive layers in LSTM
  --max_seq_len MAX_SEQ_LEN
                        Maximum sequence length of input to the LSTM
```

3. `transformer.py`:
```
usage: transformers.py [-h] [--embedding_dim EMBEDDING_DIM] [--hidden_dim HIDDEN_DIM]
                       [--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS]
                       [--learning_rate LEARNING_RATE] [--path_to_glove PATH_TO_GLOVE]
                       [--path_to_corpus PATH_TO_CORPUS] [--dropout_rate DROPOUT_RATE]
                       [--optimizer OPTIMIZER] [--scheduler SCHEDULER] [--gamma GAMMA]
                       [--use_wandb USE_WANDB] [--nhead NHEAD] [--num_decoder_layers NUM_DECODER_LAYERS]
                       [--dim_feedforward DIM_FEEDFORWARD] [--max_seq_len MAX_SEQ_LEN]

options:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Embedding Dimension
  --hidden_dim HIDDEN_DIM
                        Dimension of each Hidden Layer
  --batch_size BATCH_SIZE
                        Batch size
  --num_epochs NUM_EPOCHS
                        Number of Epochs
  --learning_rate LEARNING_RATE
                        Initial Learning Rate
  --path_to_glove PATH_TO_GLOVE
                        Path to pretrained GloVe embeddings
  --path_to_corpus PATH_TO_CORPUS
                        Path to the corpus (.txt file)
  --dropout_rate DROPOUT_RATE
                        Dropout rate for preventing overfitting
  --optimizer OPTIMIZER
                        Type of Optimizer
  --scheduler SCHEDULER
                        Scheduler for scheduling learning rate
  --gamma GAMMA         Gamma value for ExponentialLR scheduler
  --use_wandb USE_WANDB
                        Enable wandb logging
  --nhead NHEAD         Number of heads in multi-head attention
  --num_decoder_layers NUM_DECODER_LAYERS
                        Number of sub-decoder-layers in the decoder
  --dim_feedforward DIM_FEEDFORWARD
                        Dimension of the feedforward network model (Typically 4 times embedding_dim in
                        Transformer architecture)
  --max_seq_len MAX_SEQ_LEN
                        Maximum sequence length of input to the Transformer
```

## Report:
Report.pdf


----
