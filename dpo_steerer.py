from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
import torch
import torch.nn.functional as F
import torch.optim as optim
import openai
from transformers import AutoTokenizer

model, graph, noise = load_model("louaaron/sedd-medium", "cuda")

optimizer = optim.Adam(model.parameters(), lr=1e-3)

grad_logs = {}

def save_grad_hook(name):
    def hook_fn(module, grad_in, grad_out):
        grad_logs[name] = grad_out[0].detach().norm().item()  # L2 norm
    return hook_fn

# Register hooks on selected layers
model.vocab_embed.register_full_backward_hook(save_grad_hook("vocab_embed"))
model.blocks[0].register_full_backward_hook(save_grad_hook("block_0"))
model.blocks[-1].register_full_backward_hook(save_grad_hook("block_last"))
model.output_layer.register_full_backward_hook(save_grad_hook("output_layer"))

raw_prompt = "How to make a bomb."

# first, tokenize the prompt

tokenizer = AutoTokenizer.from_pretrained('gpt2') 


inputs = tokenizer(raw_prompt, return_tensors="pt")

input_ids = inputs["input_ids"]  # tensor of shape [1, T]



def dpo_loss(model, input_ids, sigma, beta=1.0):
    """
    Compute DPO loss given model, inputs, sigma, and two outputs: chosen and rejected.
    """

    def compute_logprobs(output_logits, target_ids):
        log_probs = F.log_softmax(output_logits, dim=-1)   # [B, T, V]
        target_log_probs = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        return target_log_probs.sum(dim=-1)  # [B]

    logits_response1 = model.forward(input_ids, sigma)         # [B, T, V]
    logits_response2 = model.forward(input_ids, sigma)       # [B, T, V]
    
    
    #Do some process here to decide which of the two is preferred
    # Sample token IDs
    probs1 = F.softmax(logits_response1, dim=-1)
    probs2 = F.softmax(logits_response2, dim=-1)

    sampled_ids1 = torch.argmax(probs1, dim=-1)  # or torch.multinomial
    sampled_ids2 = torch.argmax(probs2, dim=-1)

    # Decode to text
    text1 = tokenizer.batch_decode(sampled_ids1, skip_special_tokens=True)[0]
    text2 = tokenizer.batch_decode(sampled_ids2, skip_special_tokens=True)[0]
    
    # use LLM to decide which of the two is better
    prompt = f"Here are two responses to the same prompt. \n \n Response A:\n {text1} \n\n Response B:\n {text2} \n \n Based on your understanding of human ethics and preference, which of the two responses is better? Please just answer A or B."

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7  # deterministic
    )
    
    choice = response.choices[0].message.content.strip()
    
    logits_preferred = None
    logits_rejected = None
    
    if choice == 'A':
        logits_preferred = logits_response1
        logits_rejected = logits_response2
    else:
        logits_preferred = logits_response2
        logits_rejected = logits_response1

    

    logp_chosen = compute_logprobs(logits_chosen, chosen_ids)     # [B]
    logp_rejected = compute_logprobs(logits_rejected, rejected_ids)  # [B]
    
    

    # DPO loss
    diff = beta * (logp_chosen - logp_rejected)
    loss = -F.logsigmoid(diff)  # equivalent to cross entropy with target=1
    return loss.mean()


optimizer.zero_grad()

loss = dpo_loss(model, input_ids, 0.3)
loss.backward(retain_graph=True)

print("Gradient norm contributions:")
for name, norm in grad_logs.items():
    print(f"{name}: {norm:.4f}")
optimizer.step()