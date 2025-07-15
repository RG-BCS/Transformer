import time
import torch

def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train_transformer(transformer_model, train_dl, num_epochs, loss_fn, optimizer, scheduler,
                      device, input_lang, output_lang,
                      clip_norm=False, max_norm=1.0, MAX_LENGTH=20, SOS_token=1, EOS_token=2):
    transformer_model.train()
    total_loss = []
    start = time.time()
    
    for epoch in range(num_epochs):
        batch_loss = 0.0
        for eng, fre_input, fre_target, eng_length, fre_length in train_dl:
            eng, fre_input, fre_target = eng.to(device), fre_input.to(device), fre_target.to(device)
            eng_length, fre_length = eng_length.to(device), fre_length.to(device)

            result_report = transformer_model(
                eng, eng_length, fre_input, fre_length,
                causal_encoder=False, causal_decoder=True,
                enc_return_attn_weights=False, dec_return_attn_weights=False
            )

            pred = result_report['output']
            loss = loss_fn(pred.reshape(-1, pred.shape[-1]), fre_target.reshape(-1))
            loss.backward()

            if clip_norm:
                torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=max_norm)

            norm_grad = grad_norm(transformer_model)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            batch_loss += loss.item() * eng.size(0)
        
        batch_loss /= len(train_dl.dataset)
        total_loss.append(batch_loss)

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            elapsed_time = (time.time() - start) / 60
            print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {batch_loss:.4f} | Grad norm: {norm_grad:.4f} | Time: {elapsed_time:.2f}min")
            
            sample_input = "Go straight".lower()
            translated = transformer_model.generate(
                sample_input, input_lang, output_lang,
                max_len=MAX_LENGTH, SOS_token=SOS_token, EOS_token=EOS_token
            )[0]
            print(f"\nSample translation -> '{sample_input}' => '{translated}'")

            reference = "ve recto"
            print(f"Reference => '{reference}'\n")
            start = time.time()

    return total_loss
