import torch
import matplotlib.pyplot as plt
import torch.nn as nn

# Import your model and utility functions
from model import Sentiment_Model
from utils import train_transformer_encoder, plot_confusion_matrix, predict_sentiment
from dataloader_generator import train_dl, valid_dl, vocab, tokenizer, SEED

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)  # For reproducibility

# Hyperparameters and model configuration
NUM_EPOCHS = 10
num_layers = 2
src_vocab_size = len(vocab)
embed_size= 128
d_out_n_heads = embed_size
num_heads = 4
ffn_hidden_dim = 2*embed_size
dropout=0.4

# Initialize the model, loss function, and optimizer
model = Sentiment_Model(num_layers, src_vocab_size, embed_size, d_out_n_heads, num_heads, ffn_hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# Train the model
train_acc, train_loss, valid_acc, valid_loss = train_transformer_encoder(
    model, loss_fn, optimizer, train_dl, valid_dl, NUM_EPOCHS
)

# Plot training and validation accuracy and loss
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(train_acc, color='blue', label='train_acc')
plt.plot(valid_acc, color='red', label='valid_acc')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_loss, color='blue', label='train_loss')
plt.plot(valid_loss, color='red', label='valid_loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot confusion matrix on validation set
plot_confusion_matrix(model, valid_dl)
plt.show()

# Demo reviews for sentiment prediction
reviews = [
    "I absolutely loved this movie! The story was compelling, the acting was top-notch, and the soundtrack gave me chills. I’d definitely watch it again.",
    "This was a total waste of time. The plot made no sense, the characters were dull, and the ending was painfully predictable.",
    "The film had some strong performances and great cinematography, but it was dragged down by a slow-paced and confusing storyline.",
    "I am really not sure if i like or hate the movie. It was long and i honestly did not get the whole theme or plot of the movie"
]

# Predict and print sentiment for each demo review
for i, review in enumerate(reviews, 1):
    sentiment, score = predict_sentiment(model, review, vocab, tokenizer)
    print(f"Review {i} - Predicted Sentiment: {sentiment} (Confidence: {score:.4f})")
