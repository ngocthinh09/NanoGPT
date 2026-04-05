import torch
import torch.nn.functional as F
import tiktoken
import argparse
import os
from model.transformer import GPT
from config import GPTConfig, SamplingConfig

torch.serialization.add_safe_globals([GPTConfig])

def main():
    parser = argparse.ArgumentParser(description="NanoGPT Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the .pt checkpoint file')
    args = parser.parse_args()
    
    sampling_config = SamplingConfig()
    SEED = 1337
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {args.checkpoint}")
    
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    model = GPT(config)
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(sampling_config.prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(sampling_config.num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device).manual_seed(SEED)
    
    print(f"\nPrompt: {sampling_config.prompt}")
    print("-" * 50)

    while xgen.size(1) < sampling_config.max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(xgen) 
            logits = logits[:, -1, :] / sampling_config.temperature
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=20, dim=-1)
            ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)
            xcol = torch.gather(topk_indices, dim=-1, index=ix)
            xgen = torch.cat((xgen, xcol), dim=1)

    for i in range(sampling_config.num_return_sequences):
        tokens_list = xgen[i, :sampling_config.max_length].tolist()
        decoded = enc.decode(tokens_list)
        print(f"=== Sample {i+1} ===\n{decoded}\n")

if __name__ == "__main__":
    main()