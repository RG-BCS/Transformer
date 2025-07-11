"""
demo_script.py

This script trains a Transformer-based emotion classification model
on the GoEmotions dataset, evaluates it on test examples, and visualizes performance.

Author: [Your Name]
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import EmotionsModel
from utils import train_transformer_encoder, predict_from_text_or_dataset
from dataset import text_processor, train_dl, valid_dl, test_ds, dataset  # Assume dataset components are modularized

# Reproducibility
SEED = 25
torch.manual_seed(SEED)

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
num_layers = 2
src_vocab_size = len(text_processor.vocab)
embed_size = 128
d_out_n_heads = embed_size
num_heads = 4
ffn_hidden_dim = 4 * embed_size
dropout = 0.2
learning_rate = 3e-4

# Initialize model
model = EmotionsModel(
    num_layers,
    src_vocab_size,
    embed_size,
    d_out_n_heads,
    num_heads,
    ffn_hidden_dim
).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def compute_class_weights(dataset):
    """
    Computes class-wise positive weights for BCEWithLogitsLoss
    based on label frequency.

    Args:
        dataset (DatasetDict): Hugging Face dataset object.

    Returns:
        Tensor: Weights for each class.
    """
    label_freq = torch.zeros(28)
    for split in ['train', 'validation']:
        for sample in dataset[split]:
            for label in sample['labels']:
                label_freq[label] += 1
    total = label_freq.sum()
    pos_weight = total / (label_freq + 1e-6)  # Avoid division by zero
    return pos_weight


# Weighted loss to handle class imbalance
pos_weight = compute_class_weights(dataset).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Training the model
NUM_EPOCHS = 10
train_metrics_history, train_loss_history, valid_metrics_history, valid_loss_history = train_transformer_encoder(
    model, loss_fn, optimizer, train_dl, valid_dl, NUM_EPOCHS=NUM_EPOCHS
)

# ---------------------- #
# Example Inference Run
# ---------------------- #
print("\nSample text used to test model after training\n")
sample_texts = [
    "I am so happy and excited about this!",
    "This makes me really angry and sad.",
    "I'm feeling a bit anxious but hopeful.",
    "I'm feeling very sad but also relieved."
]

for i, text in enumerate(sample_texts, 1):
    emotions, confidences = predict_from_text_or_dataset(
        model, text, text_processor, device=device, threshold=0.85
    )
    print(f"Text {i}: {text}\n→ Predicted emotions: {emotions}\n")

# ---------------------- #
# Random Predictions from Test Set
# ---------------------- #
predict_from_text_or_dataset(
    model, test_ds, text_processor, n=10, threshold=0.55
)

# ---------------------- #
# Visualization
# ---------------------- #
plt.figure(figsize=(10, 5))
plt.plot([m['f1_macro'] for m in train_metrics_history], label="Train F1 Macro")
plt.plot([m['f1_macro'] for m in valid_metrics_history], label="Valid F1 Macro")
plt.title("Macro F1 Score Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("F1 Macro")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
