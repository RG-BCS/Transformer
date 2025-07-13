import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_text_sample(model, idx, max_new_tokens,
                         methods='greedy', top_k=None, temperature=0.0, eos_id=None):
    """
    Generate text tokens from the model starting from idx using various sampling methods.
    
    Args:
        model: The language model (should output logits).
        idx: Tensor of shape (1, seq_len) containing input token ids.
        max_new_tokens: Number of tokens to generate.
        methods: Sampling method: 'greedy', 'multinomial', 'temperature_sampling', 'top_k_sampling'.
        top_k: Number of top logits to consider for top_k_sampling.
        temperature: Temperature parameter for sampling (must be > 0 for temperature_sampling).
        eos_id: Token id to stop generation if generated.
        
    Returns:
        Tensor of generated tokens including the prompt idx.
    """
    idx = idx.to(device)    
    for _ in range(max_new_tokens):
        input_lengths = torch.tensor([idx.shape[1]]).to(device)
        with torch.no_grad():
            logits = model(idx, input_lengths)  # (batch=1, seq_len, vocab_size)
        logits = logits[:, -1, :]  # logits for last token: (1, vocab_size)

        if methods == 'greedy':
            next_idx = torch.argmax(torch.softmax(logits, dim=-1), dim=-1, keepdim=True)

        elif methods == 'multinomial':
            probas = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probas, num_samples=1)

        elif methods == 'temperature_sampling':
            if temperature <= 0.0:
                raise ValueError("Temperature must be > 0 for temperature_sampling")
            scaled_logits = logits / temperature
            probas = torch.softmax(scaled_logits, dim=-1)
            next_idx = torch.multinomial(probas, num_samples=1)

        elif methods == 'top_k_sampling':
            if top_k is None:
                raise ValueError("top_k must be set for top_k_sampling")
            top_logits, _ = logits.topk(top_k)
            threshold = top_logits[:, -1].unsqueeze(1)
            mask = logits < threshold
            logits = logits.masked_fill(mask, float('-inf'))
            scaled_logits = logits / (temperature if temperature > 0 else 1.0)
            probas = torch.softmax(scaled_logits, dim=-1)
            next_idx = torch.multinomial(probas, num_samples=1)

        else:
            raise ValueError(f"Unknown sampling method: {methods}")

        if eos_id is not None and next_idx.item() == eos_id:
            break

        idx = torch.cat((idx, next_idx), dim=-1)
    return idx

def grad_norm(model):
    """
    Calculate the total L2 norm of gradients for all model parameters.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train_tiny_gpt(transformer_model, train_dl, num_epochs, loss_fn, optimizer,
                   clip_norm=False, max_norm=1.0, tokenizer=None, EOS_token=None):
    """
    Train the GPT model on TinyStories dataset.
    
    Args:
        transformer_model: The model to train.
        train_dl: DataLoader for training data.
        num_epochs: Number of epochs.
        loss_fn: Loss function (e.g. CrossEntropyLoss).
        optimizer: Optimizer (e.g. Adam).
        clip_norm: Whether to clip gradients.
        max_norm: Max norm value for gradient clipping.
        tokenizer: Tokenizer to decode samples for printing.
        EOS_token: Token id for EOS token, used to stop sample generation.
        
    Returns:
        List of average losses per epoch.
    """
    transformer_model.to(device)
    transformer_model.train()
    total_loss = []
    start = time.time()

    for epoch in range(num_epochs):
        batch_loss = 0.0
        for step, (x_batch, y_batch, input_lengths) in enumerate(train_dl):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            input_lengths = input_lengths.to(device)

            pred = transformer_model(x_batch, input_lengths)

            # Check for logit explosion
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print(f"Logits contain NaN or Inf at step {step}")
                print("Stats: ", {
                    "min": pred.min().item(),
                    "max": pred.max().item(),
                    "mean": pred.mean().item(),
                    "std": pred.std().item(),
                })
                return total_loss

            loss = loss_fn(pred.reshape(-1, pred.shape[-1]), y_batch.reshape(-1))
            loss.backward()

            if clip_norm:
                torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=max_norm)

            norm_grad = grad_norm(transformer_model)

            optimizer.step()
            optimizer.zero_grad()

            batch_loss += loss.item() * y_batch.size(0)

        batch_loss /= len(train_dl.dataset)
        total_loss.append(batch_loss)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            elapsed_time = (time.time() - start) / 60
            print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {batch_loss:.4f} | Grad norm: {norm_grad:.4f} | Time: {elapsed_time:.2f}min")

            if tokenizer is not None and EOS_token is not None:
                sample_input = "how are you?"
                encode = tokenizer.encode(sample_input)
                encode_tensor = torch.tensor(encode).unsqueeze(0)
                generated = generate_text_sample(transformer_model, encode_tensor, max_new_tokens=6, eos_id=EOS_token).detach()
                print(f"\nSample input -> '{sample_input}' => '{tokenizer.decode(generated[0])}'")
            start = time.time()

    return total_loss
