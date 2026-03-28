import torch
import torch.nn as nn
from torch.nn import functional as F
from config import GPTConfig
from model.transformer import GPT
from data.loader import DataLoaderLite
import time
import math

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 100

def get_lr(it):
    # 1. Linear Warmup from 0 to max_lr for the first warmup_steps
    if (it < warmup_steps):
        return max_lr * (it + 1) / warmup_steps
    # 2. If it > warmup_steps, then decay it with cosine decay down to min_lr over the course of max_steps
    if (it > max_steps):
        return min_lr
    # 3. In between, use cosine decay from max_lr to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1, "Decay ratio should be between 0 and 1"
    coef = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coef * (max_lr - min_lr)


if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    
    B, T = 8, 512
    train_loader = DataLoaderLite(B, T)
    
    model = GPT(GPTConfig())
    model.to(device)
    model = torch.compile(model)
    
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
    
    for step in range(max_steps):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = train_loader.B * train_loader.T / (t1 - t0)
        print(f"Step: {step}| Loss: {loss.item()}| LR: {lr:.2e}| Grad Norm: {norm.item():.2f}| Time: {dt:.2f} ms| Tokens/sec: {tokens_per_sec:.2f}")