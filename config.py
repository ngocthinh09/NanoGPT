from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE tokens + 256 bytes token + 1 <|endoftext|> token
    n_layer: int = 12       # number of layers
    n_head: int = 12        # number of heads
    n_embd: int = 768       # embedding dimensions
    
@dataclass
class LRSchedulerConfig:
    max_lr: float = 6e-4        # peak learning rate
    min_lr: float = 6e-5        # minimum learning rate (after decay)
    weigh_decay: float = 0.1    # weight decay for AdamW optimizer
    
@dataclass
class TrainingConfig:
    total_batch_size: int = 524288      # total batch size across all devices and gradient accumulation steps
    B: int = 4                          # micro-batch size per device
    T: int = 1024                       # sequence length
    use_torch_compile: bool = False     # whether to use torch.compile for the model
    warmup_steps: int = 21              # number of steps to warm up the learning rate
    max_steps: int = 381                # total number of training steps

@dataclass
class SamplingConfig:
    temperature: float = 0.7            # sampling temperature
    num_return_sequences: int = 4       # number of sequences to generate
    max_length: int = 32                # maximum length of generated sequences
    prompt: str = "Once upon a time"    # initial prompt for generation
    