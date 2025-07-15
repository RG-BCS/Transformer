"""
model.py

This file implements a complete Transformer model for Neural Machine Translation (NMT) from scratch.
The model supports:
- Multi-head self-attention with masking and padding
- Positional encoding
- Encoder and decoder stacks
- Generation mode for inference
"""

import math
import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbeddingLayer(nn.Module):
    """
    Embedding + Positional Encoding Layer for input sequences.
    Applies token embeddings, positional encoding, and optional layer normalization.
    """
    def __init__(self, vocab_size, embed_size, dropout=0.1, max_len=50, PAD_token=0):
        super(EmbeddingLayer, self).__init__()
        assert embed_size % 2 == 0, "embed_size should be even number for positional encoding to work"
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_token)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_size)

        # Precompute fixed positional encodings
        pos = torch.arange(0, max_len).unsqueeze(1)
        i = torch.arange(0, embed_size, 2).float()
        denom = torch.pow(10000, i / embed_size)
        angle_rates = pos / denom
        pe = torch.zeros(max_len, embed_size)
        pe[:, 0::2] = torch.sin(angle_rates)
        pe[:, 1::2] = torch.cos(angle_rates)
        self.register_buffer('positional_encoding', pe.unsqueeze(0))

    def forward(self, x, input_lengths=None, pre_norm=False):
        seq_len = x.size(1)
        pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)
        output_ = self.dropout(self.embedding(x) * (self.embed_size) ** 0.5 + pos_enc)

        if input_lengths is not None:
            embed_mask = (torch.arange(x.size(1), device=x.device)[None, :] < input_lengths[:, None]).unsqueeze(-1)
            output_ = output_ * embed_mask

        return self.layer_norm(output_) if pre_norm else output_


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self Attention layer.
    Supports both encoder and decoder (masked) usage.
    """
    def __init__(self, embed_size, d_out_n_heads, num_heads, dropout=0.0, context_length=50, qkv_bias=False):
        super().__init__()
        assert d_out_n_heads % num_heads == 0, "d_out must be divisible by num_heads"
        self.w_key = nn.Linear(embed_size, d_out_n_heads, bias=qkv_bias)
        self.w_query = nn.Linear(embed_size, d_out_n_heads, bias=qkv_bias)
        self.w_value = nn.Linear(embed_size, d_out_n_heads, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out_n_heads, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = d_out_n_heads // num_heads
        self.register_buffer('mask', torch.triu(torch.ones((context_length, context_length)), diagonal=1))

    def forward(self, x, input_lengths=None, causal=False):
        batch, seq_len, dim_in = x.shape
        query = self.w_query(x)
        key = self.w_key(x)
        value = self.w_value(x)

        query = query.view(batch, self.num_heads, seq_len, self.head_dim)
        key = key.view(batch, self.num_heads, seq_len, self.head_dim)
        value = value.view(batch, self.num_heads, seq_len, self.head_dim)
        d_k = math.sqrt(self.head_dim)
        alignment_score = torch.matmul(query, key.transpose(-2, -1)) / d_k

        pad_mask = None
        if input_lengths is not None:
            pad_mask = torch.arange(seq_len, device=x.device)[None, :] >= input_lengths[:, None]
            pad_mask = pad_mask[:, None, None, :]

        causal_mask = None
        if causal:
            causal_mask = self.mask[:seq_len, :seq_len].bool().to(x.device)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        if pad_mask is not None:
            alignment_score = alignment_score.masked_fill(pad_mask, 1e-10)
        if causal_mask is not None:
            alignment_score = alignment_score.masked_fill(causal_mask, 1e-10)

        attention_scores = torch.softmax(alignment_score, dim=-1)
        attention_scores = self.dropout(attention_scores)

        context_vec = torch.matmul(attention_scores, value).contiguous().view(batch, seq_len, -1)

        if input_lengths is not None:
            out = self.out_proj(context_vec)
            query_mask = torch.arange(seq_len, device=x.device)[None, :] < input_lengths[:, None]
            query_mask = query_mask.unsqueeze(-1)
            output = out * query_mask
        else:
            output = self.out_proj(context_vec)

        output_skip = self.dropout(output) + x
        return output_skip, attention_scores


class DecoderCrossAttention(nn.Module):
    """
    Cross-Attention layer for decoder-to-encoder attention.
    """
    def __init__(self, embed_size, d_out_n_heads, num_heads, dropout=0.1, context_length=50, qkv_bias=False):
        super().__init__()
        assert d_out_n_heads % num_heads == 0
        self.w_key = nn.Linear(embed_size, d_out_n_heads, bias=qkv_bias)
        self.w_query = nn.Linear(embed_size, d_out_n_heads, bias=qkv_bias)
        self.w_value = nn.Linear(embed_size, d_out_n_heads, bias=qkv_bias)
        self.norm_query = nn.LayerNorm(embed_size)
        self.out_proj = nn.Linear(d_out_n_heads, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = d_out_n_heads // num_heads

    def forward(self, query, key, value, query_lengths=None, key_lengths=None, causal=False):
        assert not causal, "Causal masking is not used in cross-attention."
        batch, seq_len_q, dim_in = query.shape
        query = self.norm_query(query)
        query_input = query

        query = self.w_query(query)
        key = self.w_key(key)
        value = self.w_value(value)

        query = query.view(batch, self.num_heads, seq_len_q, self.head_dim)
        key = key.view(batch, self.num_heads, -1, self.head_dim)
        value = value.view(batch, self.num_heads, -1, self.head_dim)

        d_k = math.sqrt(self.head_dim)
        alignment_score = torch.matmul(query, key.transpose(-2, -1)) / d_k

        if key_lengths is not None:
            kv_mask = torch.arange(key.size(2), device=query.device)[None, :] >= key_lengths[:, None]
            kv_mask = kv_mask[:, None, None, :].expand(batch, self.num_heads, seq_len_q, -1)
            alignment_score = alignment_score.masked_fill(kv_mask, 1e-10)

        attn_weights = torch.softmax(alignment_score, dim=-1)
        attn_output = torch.matmul(self.dropout(attn_weights), value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len_q, -1)
        output = self.out_proj(attn_output)

        if query_lengths is not None:
            q_mask = torch.arange(seq_len_q, device=query.device)[None, :] < query_lengths[:, None]
            output = output * q_mask.unsqueeze(-1)

        output_skip = self.dropout(output) + query_input
        return output_skip, attn_weights


class FeedForward(nn.Module):
    """
    Position-wise FeedForward layer.
    """
    def __init__(self, embed_size, ffn_hidden_dim, dropout=0.0):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_hidden_dim, embed_size)
        )
        self.pre_ffn_norm = nn.LayerNorm(embed_size)
        self.post_ffn_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_lengths=None):
        batch, seq_len, _ = x.shape
        norm_x = self.pre_ffn_norm(x)
        output_ = self.ffn(norm_x)

        if input_lengths is not None:
            mask = torch.arange(seq_len, device=x.device)[None, :] < input_lengths[:, None]
            output_ = output_ * mask.unsqueeze(-1)

        return self.post_ffn_norm(self.dropout(output_) + x)


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer:
    - Self-attention + FFN
    """
    def __init__(self, embed_size, d_out_n_heads, num_heads, ffn_hidden_dim,
                 dropout=0.1, context_length=50, qkv_bias=False):
        super().__init__()
        self.mha_attn = MultiHeadAttention(embed_size, d_out_n_heads, num_heads, dropout, context_length, qkv_bias)
        self.ffn = FeedForward(embed_size, ffn_hidden_dim, dropout)

    def forward(self, x, input_lengths, causal=False, return_attn_weights=False):
        mha_output, attn_weights = self.mha_attn(x, input_lengths, causal=causal)
        ffn_output = self.ffn(mha_output, input_lengths)

        return (ffn_output, attn_weights) if return_attn_weights else ffn_output


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer:
    - Masked self-attention + Cross-attention + FFN
    """
    def __init__(self, embed_size, d_out_n_heads, num_heads, ffn_hidden_dim,
                 dropout=0.1, context_length=50, qkv_bias=False):
        super().__init__()
        self.masked_mha_attn = MultiHeadAttention(embed_size, d_out_n_heads, num_heads, dropout, context_length, qkv_bias)
        self.cross_attn = DecoderCrossAttention(embed_size, d_out_n_heads, num_heads, dropout, context_length, qkv_bias)
        self.ffn = FeedForward(embed_size, ffn_hidden_dim, dropout)

    def forward(self, target, target_lengths, key, value, key_lengths, causal=True, return_attn_weights=False):
        masked_mha_output, masked_attn_weights = self.masked_mha_attn(target, target_lengths, causal=causal)
        cross_attn_output, cross_attn_weights = self.cross_attn(masked_mha_output, key, value,
                                                                 query_lengths=target_lengths,
                                                                 key_lengths=key_lengths)
        ffn_output = self.ffn(cross_attn_output, target_lengths)

        if return_attn_weights:
            return ffn_output, masked_attn_weights, cross_attn_weights
        else:
            return ffn_output


class EncoderBlocks(nn.Module):
    """
    Stack of Encoder Layers + Embedding Layer
    """
    def __init__(self, num_layers, src_vocab_size, embed_size, d_out_n_heads, num_heads, ffn_hidden_dim,
                 dropout=0.1, context_length=50, qkv_bias=False, PAD_token=0):
        super().__init__()
        self.embedding = EmbeddingLayer(src_vocab_size, embed_size, dropout, context_length, PAD_token)
        self.layers = nn.ModuleList([
            EncoderLayer(embed_size, d_out_n_heads, num_heads, ffn_hidden_dim,
                         dropout, context_length, qkv_bias)
            for _ in range(num_layers)
        ])

    def forward(self, x, input_lengths, causal=False, return_attn_weights=False):
        embed = self.embedding(x, input_lengths)

        if return_attn_weights:
            attn_weights = []
            for layer in self.layers:
                embed, attn = layer(embed, input_lengths, causal=causal, return_attn_weights=True)
                attn_weights.append(attn)
            return embed, attn_weights

        for layer in self.layers:
            embed = layer(embed, input_lengths, causal=causal)
        return embed


class DecoderBlocks(nn.Module):
    """
    Stack of Decoder Layers + Embedding Layer
    """
    def __init__(self, num_layers, target_vocab_size, embed_size, d_out_n_heads, num_heads, ffn_hidden_dim,
                 dropout=0.1, context_length=50, qkv_bias=False, PAD_token=0):
        super().__init__()
        self.embedding = EmbeddingLayer(target_vocab_size, embed_size, dropout, context_length, PAD_token)
        self.layers = nn.ModuleList([
            DecoderLayer(embed_size, d_out_n_heads, num_heads, ffn_hidden_dim,
                         dropout, context_length, qkv_bias)
            for _ in range(num_layers)
        ])

    def forward(self, target, target_lengths, key, value, key_lengths, causal=True, return_attn_weights=False):
        embed = self.embedding(target, target_lengths)

        if return_attn_weights:
            masked_attn_weights, cross_attn_weights = [], []
            for layer in self.layers:
                embed, mask_attn, cross_attn = layer(embed, target_lengths, key, value, key_lengths,
                                                     causal=causal, return_attn_weights=True)
                masked_attn_weights.append(mask_attn)
                cross_attn_weights.append(cross_attn)
            return embed, masked_attn_weights, cross_attn_weights

        for layer in self.layers:
            embed = layer(embed, target_lengths, key, value, key_lengths, causal=causal)
        return embed


class Transformer(nn.Module):
    """
    Full Transformer for Neural Machine Translation
    Includes:
    - Encoder
    - Decoder
    - Output projection layer
    - Generation (greedy decoding) method
    """
    def __init__(self, num_layers, src_vocab_size, target_vocab_size, embed_size, d_out_n_heads, num_heads,
                 ffn_hidden_dim, dropout=0.1, context_length=500, qkv_bias=False, PAD_token=0):
        super().__init__()
        self.embed_size = embed_size
        self.encoder = EncoderBlocks(num_layers, src_vocab_size, embed_size, d_out_n_heads, num_heads,
                                     ffn_hidden_dim, dropout, context_length, qkv_bias, PAD_token)
        self.decoder = DecoderBlocks(num_layers, target_vocab_size, embed_size, d_out_n_heads, num_heads,
                                     ffn_hidden_dim, dropout, context_length, qkv_bias, PAD_token)
        self.output = nn.Linear(embed_size, target_vocab_size)

        # Optional weight tying
        if self.output.weight.shape == self.decoder.embedding.embedding.weight.shape:
            self.output.weight = self.decoder.embedding.embedding.weight

    def forward(self, src, src_lengths, target, target_lengths,
                causal_encoder=False, causal_decoder=True,
                enc_return_attn_weights=False, dec_return_attn_weights=False):
        enc_attn_weights, masked_attn_weights, cross_attn_weights = None, None, None

        if enc_return_attn_weights:
            encoder_output, enc_attn_weights = self.encoder(src, src_lengths, causal=causal_encoder,
                                                            return_attn_weights=True)
        else:
            encoder_output = self.encoder(src, src_lengths, causal=causal_encoder)

        if dec_return_attn_weights:
            decoder_output, masked_attn_weights, cross_attn_weights = self.decoder(
                target, target_lengths, encoder_output, encoder_output, src_lengths,
                causal=causal_decoder, return_attn_weights=True)
        else:
            decoder_output = self.decoder(target, target_lengths, encoder_output, encoder_output, src_lengths)

        output = self.output(decoder_output)
        return {
            'output': output,
            'encoder_attn_weights': enc_attn_weights,
            'decoder_masked_attn_weights': masked_attn_weights,
            'decoder_cross_attn_weights': cross_attn_weights
        }

    def decode_source(self, src, src_lengths, causal_encoder=False, enc_return_attn_weights=False):
        """
        Encode the source sentence and return encoder outputs.
        """
        if enc_return_attn_weights:
            return self.encoder(src, src_lengths, causal=causal_encoder, return_attn_weights=True)
        return self.encoder(src, src_lengths, causal=causal_encoder)

    def generate(self, sentence, input_lang, output_lang, max_len=20,
                 causal_encoder=False, enc_return_attn_weights=False,
                 causal_decoder=True, dec_return_attn_weights=False,
                 SOS_token=1, EOS_token=2, UNK_token=3):
        """
        Greedy decoding to translate a sentence.
        """
        with torch.no_grad():
            eng_inds = [input_lang.word2index.get(word, UNK_token) for word in sentence.strip().split()]
            eng_tensor = torch.tensor(eng_inds, dtype=torch.long).unsqueeze(0).to(device)
            input_length_tensor = torch.tensor([len(eng_inds)], dtype=torch.long).to(device)

        if enc_return_attn_weights:
            encoder_key_value, enc_attn_weights = self.decode_source(
                eng_tensor, input_length_tensor, causal_encoder, True)
        else:
            encoder_key_value = self.decode_source(eng_tensor, input_length_tensor, causal_encoder)
            enc_attn_weights = None

        generated_tokens = torch.tensor([[SOS_token]], device=device)
        output_tokens = []
        mask_attn, cross_attn = [], []

        for _ in range(max_len - 1):
            target_length = torch.tensor([generated_tokens.shape[1]], device=device)

            if dec_return_attn_weights:
                out, mask_attn_t, cross_attn_t = self.decoder(
                    generated_tokens, target_length, encoder_key_value, encoder_key_value,
                    input_length_tensor, causal=causal_decoder, return_attn_weights=True)
                mask_attn.append(mask_attn_t)
                cross_attn.append(cross_attn_t)
            else:
                out = self.decoder(generated_tokens, target_length, encoder_key_value, encoder_key_value,
                                   input_length_tensor, causal=causal_decoder)

            token_logits = self.output(out[:, -1, :])
            next_token = torch.argmax(token_logits, dim=-1)
            if next_token.item() == EOS_token:
                break

            output_tokens.append(next_token.item())
            generated_tokens = torch.cat((generated_tokens, next_token.unsqueeze(0)), dim=1)

        return {
            'output': output_tokens,
            'encoder_attn_weights': enc_attn_weights,
            'decoder_masked_attn_weights': mask_attn,
            'decoder_cross_attn_weights': cross_attn
        }
