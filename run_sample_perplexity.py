import torch
import argparse
from load_model import load_model
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch.nn.functional as F

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import get_dataset
import losses
from model.ema import ExponentialMovingAverage

@torch.no_grad()
def calc_ppl(
    model_name="louaaron/sedd-medium",
    dataset_name="wikitext103",
    batch_size=4,
    block_size=1024,
    device="cuda",
    ema_decay=0.9999,
):
    # 1. Load SEDD model components
    model, graph, noise = load_model(model_name, device)

    # 2. Load tokenizer and dataset
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", clean_up_tokenization_spaces=True)
    tokenizer.pad_token = tokenizer.eos_token  # for padding safety

    dataset = get_dataset(
        name=dataset_name,
        mode='test',
        block_size=block_size
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # 3. Create EMA wrapper for evaluation
    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

    # 4. Wrap model and EMA in state dict for loss function
    state = {
        'model': model,
        'noise': noise,
        'ema': ema
    }

    # 5. Get score-based evaluation step function
    step_fn = losses.get_step_fn(noise, graph, train=False, optimize_fn=None, accum=1)

    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)

        # Use SEDD to get loss (score entropy loss)
        loss = step_fn(state, input_ids)  # returns scalar tensor (mean over batch)
        total_loss += loss.item()
        total_tokens += input_ids.numel()
        
    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss))

    print(f"\nApproximate PPL of {model_name} on {dataset_name}: {ppl:.2f}")
    return ppl.item()

def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate SEDD samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ppl = calc_ppl(
        model_name="louaaron/sedd-medium",
        
        # dataset_name="lambada",
        # dataset_name="wikitext2",
        dataset_name="ptb",
        # dataset_name="wikitext103",
        # dataset_name="lm1b",
        
        batch_size=args.batch_size,
        device=device,
    )

if __name__ == "__main__":
    main()
