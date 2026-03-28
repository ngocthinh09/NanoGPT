import torch
import torch.nn as nn
from torch.nn import functional as F
from config import GPTConfig
from model.transformer import GPT
from data.loader import DataLoaderLite
import time

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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    for i in range(100):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = train_loader.B * train_loader.T / (t1 - t0)
        print(f"Step {i}| Loss {loss.item()}| Grad Norm {norm.item():.2f}| Time: {dt:.2f} ms| Tokens/sec: {tokens_per_sec:.2f}")