# NanoGPT

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, professional, and highly optimized re-implementation of the **GPT-2** architecture from scratch, specifically tailored for the **FineWeb-Edu** dataset. This project serves as a technical deep-dive into the mechanics of Large Language Models (LLMs), focusing on scaling efficiency and modern deep learning best practices.

### Project Overview

Inspired by Andrej Karpathy's `build-nanogpt`, this repository goes beyond a simple tutorial. It is a production-ready training pipeline designed to handle billions of tokens while maintaining a readable and modular codebase. 

Whether you are training a 124M "Small" model or scaling up to the 1.5B "XL" variant, this project integrates the latest PyTorch features-including **Distributed Data Parallel (DDP)**, **Flash Attention**, and **torch.compile**-to extract every bit of performance from your hardware.

**Key Goals:**
* **Reproducibility:** Replicating GPT-2's performance using the high-quality FineWeb-Edu dataset.
* **Efficiency:** Maximizing throughput (Tokens/sec) on consumer and enterprise GPUs.
* **Scalability:** Seamless transition from single-GPU prototyping to multi-node distributed clusters.

## Technical Highlights (Modern Training Stack)

This implementation integrates state-of-the-art techniques used in large-scale LLM training pipelines:

* **Distributed Data Parallel (DDP):** Fully compatible with `torchrun` for multi-GPU training. Orchestrates gradient synchronization and process management across distributed nodes.
* **Kernel Optimization via `torch.compile`:** Leverages Triton-based kernels to optimize the computation graph, significantly reducing overhead and increasing throughput.
* **Memory Efficiency:**
    * **Flash Attention:** An IO-aware attention mechanism that reduces memory complexity and speeds up the core transformer bottleneck.
    * **Mixed Precision (BFloat16):** Utilizes `torch.autocast` to harness Tensor Cores on NVIDIA Ampere (RTX 30-series) and Hopper architectures.
* **Professional Optimizer Tuning:**
    * **Fused AdamW:** Accelerates parameter updates using fused CUDA kernels.
    * **Weight Decay Decoupling:** Strategically applies weight decay only to weights (excluding biases and LayerNorm parameters) to improve generalization.
* **Advanced Learning Rate Schedule:** Implements a **Cosine Decay with Warmup** strategy, starting from zero and decaying to 10% of the maximum learning rate for stable convergence.

## Dataset & Tokenization

The model is trained on **FineWeb-Edu**, a high-quality educational subset of the Common Crawl.

* **Scaling Beyond RAM:** Implements a **Data Streaming & Sharding** strategy. The script processes billions of tokens in chunks (shards), allowing training on datasets that far exceed local memory limits.
* **Optimal Tokenizer:** Uses OpenAI's `tiktoken` (GPT-2 encoding) with a refined `vocab_size` of **50,304** (a multiple of 64) for maximum GPU compute alignment.

## Project Structure

```text
├── data/               # Data processing & streaming logic (FineWeb-Edu)
├── model/              # GPT-2 Architecture & Dataclass Configs
├── utils/              # DDP setup, LR Scheduler, and Logging utilities
├── checkpoints/        # Saved model weights (.pt) - auto-managed
├── logs/               # Training logs & sampling outputs
├── train.py            # Main training entry point
└── train.sh            # One-click automation script for DDP
```

## Getting Started

#### 1. Installation
```
pip install -r requirements.txt
```

#### 2. Data Preparation

Stream and tokenize the FineWeb-Edu dataset into local shards:
```
python data/fineweb-edu.py
```
**Note:**
> You can customize the `local_dir`, `remote_name`, `shard_size`, and `max_shards` parameters to control how the dataset is processed and stored.
> You can get more information about the parameters by running:
> ```
> python data/fineweb-edu.py --help
> ```

#### 3. Training (DDP)
Run the training on 2 GPUs (depend on your system, can be adjusted) using `torchrun`:
```
torchrun --nproc_per_node=2 train.py --model_type gpt2 --use_torch_compile
```

**Note:**
> Our system features Smart Auto-Detection. You don't need to specify the model type; the script reads the configuration directly from the `.pt` file:
> ```
> python train.py --resume checkpoints/model_best.pt
> ```

## License
This project is licensed under the MIT License.