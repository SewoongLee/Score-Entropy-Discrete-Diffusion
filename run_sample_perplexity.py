import torch
import argparse
from load_model import load_model
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch.nn.functional as F
import sampling

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_perplexity(samples, eval_model):
    with torch.no_grad():
        # samples: [batch, seq_len]
        # logits: [batch, seq_len, vocab_size]
        loss, logits = eval_model(samples, labels=samples)[:2]
        logits = logits.transpose(-1, -2)  # [batch, seq_len, vocab_size] -> [batch, vocab_size, seq_len]

         # Shift for GPT2-style next-token prediction
        perplexity = F.cross_entropy(
            logits[..., :-1],  # pred (shifted right)
            samples[..., 1:],  # True (shifted left)
            reduction="none"   # per-token
        )
        perplexity = perplexity.mean(dim=-1).exp().mean()
        return perplexity


def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate SEDD samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    # parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load SEDD model
    model, graph, noise = load_model(args.model_path, device)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    # Sample text from SEDD
    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (args.batch_size, 1024), 'analytic', args.steps, device=device
    )
    samples = sampling_fn(model)  # [batch, seq_len]

    # Decode and print
    text_samples = tokenizer.batch_decode(samples)
    for i, text in enumerate(text_samples):
        print(f"[Sample {i}]")
        print(text)
        print("="*60)

    # Load GPT-2 for perplexity evaluation
    eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()

    # Calculate perplexity
    ppl = compute_perplexity(samples, eval_model)
    print(f"\nPerplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
