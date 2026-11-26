# Transformers & State Space Models (SSMs) in PyTorch

A clean, modular, and from-scratch PyTorch implementation of modern sequence modeling architectures—covering Transformer Encoders, Decoders, and State Space Models (SSMs)—alongside two applied end-to-end projects.

---

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Core Architectures Implemented](#core-architectures-implemented)
- [Projects](#projects)
  - [Project 1: Custom Large Language Model (LLM)](#project-1-custom-large-language-model-llm)
  - [Project 2: State Space Model Research Exploration](#project-2-state-space-model-research-exploration)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [License](#license)

---

## Overview

This repository serves as both an educational reference and an experimental codebase for deep learning sequence architectures. Every core layer (Multi-Head Attention, Rotary Positional Embeddings, Selective State Spaces, etc.) is built cleanly in PyTorch with minimal external abstractions so you can easily inspect tensor shapes, gradients, and training dynamics.

---

## Repository Structure

```text
├── architectures/
│   ├── transformer/
│   │   ├── attention.py        # Scaled Dot-Product, Multi-Head, & Causal Attention
│   │   ├── embeddings.py       # Sinusoidal, Learned, and RoPE (Rotary) Embeddings
│   │   ├── encoder.py          # Encoder block & full Encoder stack
│   │   ├── decoder.py          # Masked Decoder block & Cross-Attention
│   │   └── model.py            # Encoder-Decoder & Decoder-only wrappers
│   └── ssm/
│       ├── s4.py               # Structured State Space sequence model (S4)
│       ├── selective_scan.py   # Hardware-aware / pure PyTorch selective scan
│       └── mamba.py            # Mamba block & full SSM backbone
├── projects/
│   ├── 01_custom_llm/
│   │   ├── train.py            # Pre-training script with mixed precision & cosine LR
│   │   ├── generate.py         # Autoregressive text generation (Top-k, Top-p, Temperature)
│   │   └── config.yaml         # Model hyperparameters
│   └── 02_ssm_research/
│       ├── benchmark.py        # Memory & throughput scaling vs. sequence length
│       ├── long_range_eval.py  # Long-context recall & sequence modeling tasks
│       └── notes.md            # Experimental logs and findings
├── requirements.txt
└── README.md
```

---

## Core Architectures Implemented

### 1. Transformer Encoder
* Bidirectional Multi-Head Self-Attention (MHSA).
* Pre-LayerNorm (`Pre-LN`) stability and Residual Connections.
* Position-wise Feed-Forward Networks (FFN) with GELU / SwiGLU activations.

### 2. Transformer Decoder
* Causal (masked) Multi-Head Self-Attention for autoregressive modeling.
* Optional Encoder-Decoder Cross-Attention for sequence-to-sequence tasks.
* KV-Caching support for fast inference.

### 3. State Space Models (SSMs)
* Continuous-to-discrete state-space discretization (Zero-Order Hold / Bilinear).
* Linear-time $O(N)$ sequence modeling compared to the Transformer's $O(N^2)$ attention bottleneck.
* Input-dependent selection mechanisms (Selective SSM / Mamba-style gating).

---

## Projects

### Project 1: Custom Large Language Model (LLM)
An end-to-end Decoder-only Transformer trained from scratch on text data.
* **Key Features:** Custom BPE/tiktoken tokenizer integration, Rotary Position Embeddings (RoPE), SwiGLU MLP layers, gradient accumulation, and `torch.compile` support.
* **Goal:** Demonstrate full pre-training pipeline, loss curve tracking, and autoregressive text sampling.

### Project 2: State Space Model Research Exploration
An experimental benchmark and research sandbox evaluating SSMs against standard Transformers on long-sequence tasks.
* **Key Features:** Sequence-length scaling benchmarks (1k to 32k+ tokens), VRAM consumption profiling, and synthetic associative recall tests.
* **Goal:** Investigate where linear-time state-space recurrences outperform attention in speed/memory, and analyze trade-offs in long-range information retention.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd your-repo-name
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

*Core requirements: `torch >= 2.1.0`, `numpy`, `einops`, `pyyaml`, `tqdm`.*

---

## Usage

### Train the LLM (Project 1)
```bash
python projects/01_custom_llm/train.py --config projects/01_custom_llm/config.yaml
```

### Generate Text
```bash
python projects/01_custom_llm/generate.py --checkpoint checkpoints/llm_latest.pt --prompt "Once upon a time"
```

### Run SSM Benchmarks (Project 2)
```bash
python projects/02_ssm_research/benchmark.py --seq-lengths 1024 4096 16384
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
