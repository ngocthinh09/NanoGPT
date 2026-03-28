import math

class LRScheduler:
    def __init__(self, max_lr, min_lr, warmup_steps, max_steps):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, it):
        # 1. Linear Warmup from 0 to max_lr for the first warmup_steps
        if (it < self.warmup_steps):
            return self.max_lr * (it + 1) / self.warmup_steps
        # 2. If it > warmup_steps, then decay it with cosine decay down to min_lr over the course of max_steps
        if (it > self.max_steps):
            return self.min_lr
        # 3. In between, use cosine decay from max_lr to min_lr
        decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1, "Decay ratio should be between 0 and 1"
        coef = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coef * (self.max_lr - self.min_lr)