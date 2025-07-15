# Transformer Models from Scratch – NLP Projects Collection

Welcome to my repository of NLP projects built entirely from **scratch using PyTorch**, centered around the
Transformer architecture. This repo demonstrates a deep understanding of modern NLP, from tokenization to attention
mechanisms, to full model training on real-world datasets.

---

## Overview

This repository features several standalone projects that explore core NLP tasks using custom-built Transformer
models, without relying on high-level libraries like Hugging Face Transformers. Every model, including attention
layers, encoder-decoder structure, masking, and training loop, is handcrafted to promote deep architectural understanding.

---

##  Projects Included

Project so far completed:
    

| Project                   | Task Type                    | Dataset           | Highlights                                              |
|---------------------------|------------------------------|-------------------|---------------------------------------------------------|
| `sentiment_analysis/`     | Text Classification          | IMDb              | Binary classification using Transformer encoder         |
| `go_emotions_multilabel/` | Multilabel Emotion Detection | Google GoEmotions | Handles multilabel classification with threshold tuning |
| `tinygpt/`                | Autoregressive Generation    | TinyStories       | GPT-style language modeling with causal masking         |
| `nmt_eng_spa/`            | Neural Machine Translation   | Eng–Spa corpus    | Encoder–decoder Transformer with teacher forcing        |

---

## Built From Scratch

Each project includes:
- Custom Transformer modules (Multi-Head Attention, LayerNorm, FFN, Positional Encoding, etc.)
- Tokenization and data loading tailored for the task
- Training and evaluation loops with gradient clipping and learning rate schedulers
- Clean and modular PyTorch code

No external model libraries are used. Everything is implemented using `torch.nn`, `torch.optim`, and a few utility
libraries (`matplotlib`, `numpy`, etc.).

---

##  Why This Repo?

This collection is both an **educational journey** and a **practical toolkit**. It showcases:
- A thorough understanding of how Transformers work internally
- The ability to adapt the architecture across diverse NLP tasks
- Resource-efficient experiments on modest hardware

---

##  Running the Projects

Each folder is self-contained. To run a project:

```bash
cd sentiment_analysis/  # or any other project folder
python demo_script.py
