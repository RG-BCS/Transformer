import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time

def plot_confusion_matrix(model, dataloader_, labels=['Negative', 'Positive'], 
                          normalize=False, title='Confusion Matrix'):
    """
    Plots the confusion matrix for predictions made by the model on a dataloader.

    Args:
        model: Trained sentiment analysis model.
        dataloader_: DataLoader containing samples to evaluate.
        labels: Class labels to show on the matrix.
        normalize: If True, normalize the matrix.
        title: Title of the plot.
    """
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for x, y, lengths in dataloader_:
            x = x.to(next(model.parameters()).device)
            y = y.to(next(model.parameters()).device).float()
            lengths = lengths.to(next(model.parameters()).device)

            preds = model(x, lengths).squeeze(-1)
            preds = (preds >= 0.5).float()

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='.2f' if normalize else 'd')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def grad_norm(model):
    """
    Computes the total L2 norm of all gradients in the model.
    
    Args:
        model: PyTorch model with gradients.

    Returns:
        Total gradient norm (float).
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train(model, loss_fn, optimizer, dataloader):
    """
    Trains the model for one epoch.

    Args:
        model: Sentiment model.
        loss_fn: Loss function (e.g., BCE).
        optimizer: Optimizer for updating parameters.
        dataloader: Training DataLoader.

    Returns:
        Tuple of (accuracy, loss, gradient norm)
    """
    model.train()
    total_accuracy, total_loss = 0.0, 0.0

    for text_batch, label_batch, lengths in dataloader:
        text_batch, label_batch, lengths = text_batch.to(device), label_batch.to(device), lengths.to(device)
        pred = model(text_batch, lengths)[:, 0]
        loss = loss_fn(pred, label_batch)
        loss.backward()

        norm_grad = grad_norm(model)

        optimizer.step()
        optimizer.zero_grad()

        total_accuracy += ((pred >= 0.5).float() == label_batch).sum().item()
        total_loss += loss.item() * label_batch.size(0)

    return total_accuracy / len(dataloader.dataset), total_loss / len(dataloader.dataset), norm_grad

def evaluate(model, loss_fn, dataloader):
    """
    Evaluates the model on a validation/test set.

    Args:
        model: Trained sentiment model.
        loss_fn: Loss function.
        dataloader: DataLoader for validation/test.

    Returns:
        Tuple of (accuracy, loss)
    """
    model.eval()
    total_accuracy, total_loss = 0.0, 0.0

    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            text_batch, label_batch, lengths = text_batch.to(device), label_batch.to(device), lengths.to(device)
            pred = model(text_batch, lengths)[:, 0]
            loss = loss_fn(pred, label_batch)

            total_accuracy += ((pred >= 0.5).float() == label_batch).sum().item()
            total_loss += loss.item() * label_batch.size(0)

    return total_accuracy / len(dataloader.dataset), total_loss / len(dataloader.dataset)

def train_transformer_encoder(model, loss_fn, optimizer, train_dl, valid_dl, NUM_EPOCHS=10):
    """
    Runs the full training loop for the transformer encoder model.

    Args:
        model: Transformer sentiment model.
        loss_fn: Loss function (e.g., BCE).
        optimizer: Optimizer.
        train_dl: Training DataLoader.
        valid_dl: Validation DataLoader.
        NUM_EPOCHS: Number of training epochs.

    Returns:
        Tuple of train/val accuracy and loss over epochs.
    """
    train_acc, train_loss = [], []
    valid_acc, valid_loss = [], []

    for epoch in range(NUM_EPOCHS):
        start = time.time()

        acc_train, loss_train, norm_grad = train(model, loss_fn, optimizer, train_dl)
        acc_valid, loss_valid = evaluate(model, loss_fn, valid_dl)

        train_acc.append(acc_train)
        train_loss.append(loss_train)
        valid_acc.append(acc_valid)
        valid_loss.append(loss_valid)

        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Time: {elapsed:.2f}s | "
              f"Train Acc:{acc_train:.4f} | Train Loss:{loss_train:.4f} | "
              f"Valid Acc:{acc_valid:.4f} | Valid Loss:{loss_valid:.4f} | Grad norm:{norm_grad:.2f}")

    return train_acc, train_loss, valid_acc, valid_loss

def predict_sentiment(model, text, vocab, tokenizer, device='cuda'):
    """
    Predicts the sentiment of a single text input.

    Args:
        model: Trained sentiment model.
        text: Raw input text (string).
        vocab: Token-to-index mapping.
        tokenizer: Tokenizer function.
        device: 'cuda' or 'cpu'.

    Returns:
        Tuple of (sentiment label, probability)
    """
    model.eval()
    with torch.no_grad():
        tokens = tokenizer(text)
        token_ids = [vocab.get(tok, vocab['<unk>']) for tok in tokens]

        input_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
        lengths = torch.tensor([len(token_ids)]).to(device)

        output = model(input_tensor, lengths).item()
        return "Positive" if output >= 0.5 else "Negative", output

# Example inference (demo)
reviews = [
    "I absolutely loved this movie! The story was compelling, the acting was top-notch, and the soundtrack gave me chills. I’d definitely watch it again.",
    "This was a total waste of time. The plot made no sense, the characters were dull, and the ending was painfully predictable.",
    "The film had some strong performances and great cinematography, but it was dragged down by a slow-paced and confusing storyline.",
    "I am really not sure if i like or hate the movie. It was long and i honestly did not get the whole theme or plot of the movie"
]

for i, review in enumerate(reviews, 1):
    sentiment, score = predict_sentiment(model, review, vocab, tokenizer)
    print(f"Review {i} - Predicted Sentiment: {sentiment} (Confidence: {score:.4f})")
