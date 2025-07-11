"""
utils.py

Utility functions for training and evaluating a transformer-based emotion classifier.
Includes metric computation, training loops, inference/prediction utilities, and gradient diagnostics.

Author: [Your Name]
"""

import torch
import random
import time
from sklearn.metrics import precision_score, recall_score, f1_score


def grad_norm(model):
    """
    Computes the total gradient norm of all model parameters.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        float: L2 norm of gradients.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def predict_from_text_or_dataset(model, input_data, text_processor, n=5, device='cuda', threshold=0.3):
    """
    Predict emotions from raw input string or randomly sampled entries from EmotionDataset.

    Args:
        model (nn.Module): Trained model.
        input_data (str or EmotionDataset): Text string or dataset.
        text_processor (TextProcessor): Processor object for tokenization.
        n (int): Number of samples (used if input is a dataset).
        device (str): CUDA or CPU.
        threshold (float): Classification threshold.

    Returns:
        If input is str: (predicted_emotions, probabilities)
        If input is EmotionDataset: Prints results for n samples.
    """
    model.eval()
    if isinstance(input_data, EmotionDataset):
        sampled_indices = random.sample(range(len(input_data)), n)

        for i, idx in enumerate(sampled_indices, 1):
            token_ids, label_vec = input_data[idx]
            input_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
            length_tensor = torch.tensor([len(token_ids)]).to(device)

            with torch.no_grad():
                logits = model(input_tensor, length_tensor)
                probs = torch.sigmoid(logits).cpu().squeeze(0)

            predicted_indices = (probs >= threshold).nonzero(as_tuple=True)[0].tolist()
            predicted_emotions = [text_processor.vocab_emotions[idx] for idx in predicted_indices]

            decoded_words = text_processor.int_to_word(token_ids)
            decoded_text = ' '.join(decoded_words)

            label_tensor = torch.tensor(label_vec)
            true_indices = (label_tensor >= 0.5).nonzero(as_tuple=True)[0].tolist()
            true_emotions = [text_processor.vocab_emotions[idx] for idx in true_indices]

            print(f"\nSample {i}")
            print(f"Input Text       : {decoded_text}")
            print(f"True Emotions    : {true_emotions}")
            print(f"Predicted Emotions: {predicted_emotions}")
            print("-" * 60)

    else:
        # Handle single text prediction
        token_ids = text_processor.word_to_int(input_data)
        input_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
        length_tensor = torch.tensor([len(token_ids)]).to(device)

        with torch.no_grad():
            logits = model(input_tensor, length_tensor)
            probs = torch.sigmoid(logits).cpu().squeeze(0)

        predicted_indices = (probs >= threshold).nonzero(as_tuple=True)[0].tolist()
        predicted_emotions = [text_processor.vocab_emotions[idx] for idx in predicted_indices]

        return predicted_emotions, probs.tolist()


def compute_metrics(preds, targets, threshold=0.5):
    """
    Compute micro and macro precision, recall, and F1 metrics.

    Args:
        preds (Tensor): Predicted scores (after sigmoid).
        targets (Tensor): Ground-truth multi-hot labels.
        threshold (float): Binarization threshold.

    Returns:
        dict: Dictionary of micro and macro metrics.
    """
    preds_bin = (preds >= threshold).cpu().numpy()
    targets_np = targets.cpu().numpy()

    precision_macro = precision_score(targets_np, preds_bin, average='macro', zero_division=0)
    recall_macro = recall_score(targets_np, preds_bin, average='macro', zero_division=0)
    f1_macro = f1_score(targets_np, preds_bin, average='macro', zero_division=0)

    precision_micro = precision_score(targets_np, preds_bin, average='micro', zero_division=0)
    recall_micro = recall_score(targets_np, preds_bin, average='micro', zero_division=0)
    f1_micro = f1_score(targets_np, preds_bin, average='micro', zero_division=0)

    return {
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro
    }


def train(model, loss_fn, optimizer, dataloader):
    """
    One training epoch loop.

    Args:
        model (nn.Module): The model.
        loss_fn: Loss function.
        optimizer: Optimizer instance.
        dataloader (DataLoader): DataLoader with training data.

    Returns:
        Tuple: (metrics, average_loss, gradient_norm)
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for text_batch, label_batch, lengths in dataloader:
        text_batch, label_batch, lengths = text_batch.to(device), label_batch.to(device), lengths.to(device)

        optimizer.zero_grad()
        pred = model(text_batch, lengths)
        loss = loss_fn(pred, label_batch)
        loss.backward()

        norm_grad = grad_norm(model)
        optimizer.step()

        total_loss += loss.item() * label_batch.size(0)
        all_preds.append(torch.sigmoid(pred).detach().cpu())
        all_targets.append(label_batch.detach().cpu())

    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(all_preds, all_targets)
    return metrics, avg_loss, norm_grad


def evaluate(model, loss_fn, dataloader):
    """
    Evaluation loop without gradient updates.

    Args:
        model (nn.Module): The model.
        loss_fn: Loss function.
        dataloader (DataLoader): DataLoader with validation or test data.

    Returns:
        Tuple: (metrics, average_loss)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            text_batch, label_batch, lengths = text_batch.to(device), label_batch.to(device), lengths.to(device)
            pred = model(text_batch, lengths)
            loss = loss_fn(pred, label_batch)

            total_loss += loss.item() * label_batch.size(0)
            all_preds.append(torch.sigmoid(pred).cpu())
            all_targets.append(label_batch.cpu())

    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(all_preds, all_targets)
    return metrics, avg_loss


def train_transformer_encoder(model, loss_fn, optimizer, train_dl, valid_dl, NUM_EPOCHS=10):
    """
    Trains a transformer encoder over multiple epochs and evaluates on validation data.

    Args:
        model: Transformer model.
        loss_fn: Loss function.
        optimizer: Optimizer.
        train_dl: Training DataLoader.
        valid_dl: Validation DataLoader.
        NUM_EPOCHS (int): Number of epochs.

    Returns:
        Tuple: (train_metrics_list, train_loss_list, valid_metrics_list, valid_loss_list)
    """
    train_metrics, train_loss = [], []
    valid_metrics, valid_loss = [], []

    for epoch in range(NUM_EPOCHS):
        start = time.time()

        train_metric, loss_train, norm_grad = train(model, loss_fn, optimizer, train_dl)
        valid_metric, loss_valid = evaluate(model, loss_fn, valid_dl)

        train_metrics.append(train_metric)
        train_loss.append(loss_train)
        valid_metrics.append(valid_metric)
        valid_loss.append(loss_valid)

        elapsed = time.time() - start

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Time: {elapsed:.2f}s | "
              f"Train Loss: {loss_train:.4f} | Valid Loss: {loss_valid:.4f} | Grad Norm: {norm_grad:.2f}")
        print(f"Train F1 Macro: {train_metric['f1_macro']:.4f} | Valid F1 Macro: {valid_metric['f1_macro']:.4f}\n")

    return train_metrics, train_loss, valid_metrics, valid_loss
