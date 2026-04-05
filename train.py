import tiktoken
import torch
import torch.nn.functional as F
from config import GPTConfig, LRSchedulerConfig, TrainingConfig, SamplingConfig
from model.transformer import GPT
from data.loader import DataLoaderLite
from utils.lr_scheduler import LRScheduler
from utils.distributed import ddp_setup, ddp_cleanup
from utils.logger import get_logger
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import logging, time, os

sampling_config = SamplingConfig()
lr_scheduler_config = LRSchedulerConfig()
training_config = TrainingConfig()
lr_scheduler = LRScheduler(
    max_lr=lr_scheduler_config.max_lr,
    min_lr=lr_scheduler_config.min_lr,
    warmup_steps=training_config.warmup_steps,
    max_steps=training_config.max_steps
)

if __name__ == "__main__":
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = ddp_setup()
    root_logger = get_logger(name="NanoGPT", log_dir="logs", master_process=master_process)
    logger = logging.getLogger(f'NanoGPT.train')
    if master_process:
        logger.info(f'STARTING TRAINING PROCESS')
        logger.info(f"DDP World Size: {ddp_world_size}")
        
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    if master_process:
        logger.info(f"Using device: {device}")
    
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    enc = tiktoken.get_encoding("gpt2")
    
    total_batch_size = training_config.total_batch_size
    B = training_config.B
    T = training_config.T
    assert total_batch_size % (B * T * ddp_world_size) == 0, "Total batch size must be divisible by B * T * ddp_world_size"
    gradient_accumulation_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        logger.info(f"Total Desired Batch Size: {total_batch_size}")
        logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")

    train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, split='train')
    val_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, split='val')

    model = GPT(GPTConfig(vocab_size=50304)).to(device)
    if training_config.use_torch_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    original_model = model.module if ddp else model
    
    optimizer = original_model.configure_optimizers(
        weight_decay=lr_scheduler_config.weigh_decay,
        learning_rate=lr_scheduler_config.max_lr,
        master_process=master_process,
        device_type=device_type
    )
    
    for step in range(training_config.max_steps):
        t0 = time.time()
        
        # Calculate validation loss
        if step % 15 == 0:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_step = 20
                for _ in range(val_loss_step):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_step
                    val_loss_accum += loss.detach()
                if ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                if master_process:
                    logger.info(f"Validation Loss: {val_loss_accum.item():.4f}")
                    if step > 0 and (step % 30 == 0 or step == training_config.max_steps - 1):
                        os.makedirs("checkpoints", exist_ok=True)
                        checkpoint_path = os.path.join("checkpoints", f"model_step_{step}.pt")
                        checkpoint = {
                            'model': original_model.state_dict(),
                            'config': original_model.config,
                            'step': step,
                            'val_loss': val_loss_accum.item()
                        }
                        torch.save(checkpoint, checkpoint_path)

        # Sampling from the model
        if step > 0 and step % 15 == 0 and not training_config.use_torch_compile and master_process:
            model.eval()
            tokens = enc.encode(sampling_config.prompt)
            tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(sampling_config.num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device).manual_seed(1337)
            while xgen.size(1) < sampling_config.max_length:
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(xgen)
                    logits = logits[:, -1, :] / sampling_config.temperature
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, k=20, dim=-1)
                    ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, dim=-1, index=ix)
                    xgen = torch.cat((xgen, xcol), dim=1)
                    
            for i in range(sampling_config.num_return_sequences):
                tokens = xgen[i, :sampling_config.max_length].tolist()
                decoded = enc.decode(tokens)
                logger.info(f"\n=== Sample {i+1} ===\n{decoded}\n")
            
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(gradient_accumulation_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / gradient_accumulation_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = lr_scheduler.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = train_loader.B * train_loader.T * ddp_world_size * gradient_accumulation_steps / (t1 - t0)
        if master_process:
            logger.info(f"Step: {step:5d}| Loss: {loss_accum.item()}| LR: {lr:.2e}| Grad Norm: {norm.item():.2f}| Time: {dt:.2f} ms| Tokens/sec: {tokens_per_sec:.2f}")
            
    if ddp:
        ddp_cleanup()
            