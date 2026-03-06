<p align="center">
  <h1 align="center">🧠 MicroGPT — Learning Lab</h1>
  <p align="center">
    <em>Understanding GPT from the ground up by studying, dissecting, and rebuilding Karpathy's 200-line GPT.</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.8+-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/dependencies-zero-brightgreen?style=flat-square" alt="Zero Dependencies">
    <img src="https://img.shields.io/badge/status-learning%20in%20progress-orange?style=flat-square" alt="Status">
    <img src="https://img.shields.io/badge/license-MIT-purple?style=flat-square" alt="License">
  </p>
</p>

---

## 📌 What Is This?

This repo is my **hands-on learning journey** through how GPT and Transformers work — starting from Andrej Karpathy's [MicroGPT](https://github.com/karpathy/microgpt), a complete GPT implementation in **200 lines of pure Python** with zero dependencies.

Instead of just reading the code, I'm:
- 🔬 **Dissecting** every function into its simplest form
- 🧮 **Expanding** compact one-liners into step-by-step logic
- 📝 **Documenting** everything for someone who struggles with math formulas
- 🛠️ **Rebuilding** each component from scratch to truly understand it

---

## 📂 Repository Structure

```
MicroGPT/
│
├── microgpt.py                  # Original MicroGPT source (Karpathy's 200-line GPT)
├── input.txt                    # Training data (names dataset)
│
├── Scratch/
│   └── microgpt.py              # My annotated version with expanded, readable code
│
├── Functions_Explaination/
│   ├── matrix_vector_multiplication.py   # Matrix-vector multiply broken down
│   └── unique_alphabet.py                # Tokenizer logic explored
│
├── LEARNING_GUIDE.md            # Step-by-step learning guide (beginner-friendly)
└── README.md                    # You are here
```

---

## 🧩 Components Covered

### ✅ Completed

| Component | File(s) | Description |
|---|---|---|
| Original Source | `microgpt.py` | Karpathy's unmodified 200-line GPT |
| Expanded Code | `Scratch/microgpt.py` | Readable version with expanded functions |
| Matrix Multiply | `Functions_Explaination/matrix_vector_multiplication.py` | `linear()` function broken into explicit loops |
| Tokenizer | `Functions_Explaination/unique_alphabet.py` | How characters become token IDs |
| Learning Guide | `LEARNING_GUIDE.md` | 14-step beginner guide — numbers to GPT |

### 🔜 Planned

| Component | Description | Status |
|---|---|---|
| Softmax Breakdown | Step-by-step softmax with numeric examples | 📋 Planned |
| RMSNorm Breakdown | Normalization logic explored | 📋 Planned |
| Attention Deep Dive | Single-head attention traced with real numbers | 📋 Planned |
| Multi-Head Attention | How heads split and recombine | 📋 Planned |
| Autograd Exploration | Tracing the `Value` class computation graph | 📋 Planned |
| Backpropagation Walkthrough | Manual gradient computation examples | 📋 Planned |
| Adam Optimizer | Momentum, velocity, bias correction explained | 📋 Planned |
| Loss Function | Cross-entropy computed step by step | 📋 Planned |
| Training Visualizations | Loss curves, embedding visualizations | 📋 Planned |
| Custom Dataset Training | Train on city names, Pokémon, etc. | 📋 Planned |
| Comparison with nanoGPT | How MicroGPT concepts scale to PyTorch | 📋 Planned |

---

## 🚀 Quick Start

### Run the original MicroGPT
```bash
python microgpt.py
```
This will:
1. Download the names dataset (if not present)
2. Train a character-level GPT for 1000 steps
3. Generate 20 hallucinated names

### Run the expanded (scratch) version
```bash
cd Scratch
python microgpt.py
```
Same behavior, but with more readable, expanded code.

> **Note:** Training takes a few minutes on CPU. This is expected — MicroGPT uses scalar-level autograd (each `Value` wraps a single float), which is intentionally simple but slow.

---

## 📖 Learning Resources

| Resource | What It Is |
|---|---|
| [LEARNING_GUIDE.md](./LEARNING_GUIDE.md) | Step-by-step guide from zero to GPT — every concept with analogies, arithmetic, and code |

### Recommended External Resources

| Order | Resource | Link |
|---|---|---|
| 1 | 3Blue1Brown — Neural Networks | [YouTube](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) |
| 2 | Karpathy — micrograd | [YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0) |
| 3 | Karpathy — makemore series | [YouTube](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) |
| 4 | Jay Alammar — Illustrated Transformer | [Blog](https://jalammar.github.io/illustrated-transformer/) |
| 5 | Karpathy — Let's build GPT | [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY) |

---

## 🏗️ Architecture Overview

MicroGPT implements a simplified GPT-2 in 200 lines:

```
Token → [Embedding + Position] → [RMSNorm]
                                      ↓
                          ┌── Transformer Block ──┐
                          │  RMSNorm → Attention   │
                          │       ↓ + residual     │
                          │  RMSNorm → MLP         │
                          │       ↓ + residual     │
                          └────────────────────────┘
                                      ↓
                          [Linear → Logits → Softmax → Prediction]
```

| Hyperparameter | Value | Meaning |
|---|---|---|
| `n_embd` | 16 | Embedding dimension |
| `n_head` | 4 | Number of attention heads |
| `n_layer` | 1 | Number of transformer layers |
| `block_size` | 16 | Maximum sequence length |
| `vocab_size` | 27 | 26 letters + 1 BOS token |
| Total params | ~5,000 | Tiny! GPT-4 has ~1.8 trillion |

---

## 🤝 Contributing

This is a personal learning repo, but if you'd like to:
- 📝 Improve explanations
- 🧮 Add a new function breakdown in `Functions_Explaination/`
- 🐛 Fix an error in the guides
- 💡 Suggest a new exploration topic

Feel free to open an issue or PR!

---

## 📄 Credits & License

- **MicroGPT** by [Andrej Karpathy](https://github.com/karpathy) — the original 200-line implementation
- **Learning materials** created as a personal study guide

Licensed under [MIT](./LICENSE).

---

<p align="center">
  <em>"Everything you need to understand modern AI is in these 200 lines. Everything else is just efficiency."</em>
</p>
