import torch
from config import GPTConfig
from model.transformer import GPT
from data.loader import DataLoaderLite
from utils.lr_scheduler import LRScheduler
import time

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 100

lr_scheduler = LRScheduler(max_lr, min_lr, warmup_steps, max_steps)

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
    
    total_batch_size = 524288
    B = 4
    T = 1024
    assert total_batch_size % (B * T) == 0, "Total batch size must be divisible by B * T"
    gradient_accumulation_steps = total_batch_size // (B * T)
    print(f"Total Desired Batch Size: {total_batch_size}")
    print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
    train_loader = DataLoaderLite(B, T)
    
    model = GPT(GPTConfig())
    model.to(device)
    model = torch.compile(model)
    
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
    
    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(gradient_accumulation_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / gradient_accumulation_steps
            loss_accum += loss.detach()
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = lr_scheduler.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = train_loader.B * train_loader.T * gradient_accumulation_steps / (t1 - t0)
        print(f"Step: {step}| Loss: {loss_accum.item()}| LR: {lr:.2e}| Grad Norm: {norm.item():.2f}| Time: {dt:.2f} ms| Tokens/sec: {tokens_per_sec:.2f}")