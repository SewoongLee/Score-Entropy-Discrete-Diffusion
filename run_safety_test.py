import os
import json
import torch
from tqdm import tqdm
import argparse
from load_model import load_model
from transformers import GPT2TokenizerFast
import sampling
from datasets import load_dataset


device = torch.device('cuda')


def load_ds():
    dataset = load_dataset('JailbreakBench/JBB-Behaviors', 'behaviors')
    ds = []
    for datapoint in dataset['harmful']:
        ds.append({
            'prefix': f'Question: {datapoint["Goal"]}. Answer: ',
            'suffix': '',
            'behavior': datapoint["Behavior"],
            'category': datapoint["Category"],
        })
    for datapoint in dataset['benign']:
        ds.append({
            'prefix': f'Question: {datapoint["Goal"]}. Answer: ',
            'suffix': '',
            'behavior': datapoint["Behavior"],
            'category': datapoint["Category"],
        })
    return ds


def get_model_and_tokenizer(model_path):
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', clean_up_tokenization_spaces=True)
    model, graph, noise = load_model(model_path, device)
    return model, graph, noise, tokenizer


def generate_response(model, graph, noise, tokenizer, prefix, suffix, batch_size, steps):
    prefix_ids = tokenizer(prefix).input_ids
    suffix_ids = tokenizer(suffix).input_ids
    input_ids = prefix_ids + suffix_ids
    input_locs = list(range(len(prefix_ids))) + list(range(1024-len(suffix_ids), 1024))
    input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(batch_size, 1)

    def proj_fun(x):
        x[:, input_locs] = input_ids
        return x
    
    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (batch_size, 1024), 'analytic', steps, device=device, proj_fun=proj_fun
    )
    samples = proj_fun(sampling_fn(model))
    text_samples = tokenizer.batch_decode(samples)
    return text_samples[0]


def judge_responses(responses):
    # TO-DO [Aditi]
    # The prompt/question can be accessed by `responses[i]['prefix']`.
    # The response can be accessed by `responses[i]['response']`.
    # Note that the response contains the prefix as well.
    pass


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--save_response_path", type=str, default="responses.json")
    args = parser.parse_args()

    # Get dataset, model, and tokenizer
    ds = load_ds()
    model, graph, noise, tokenizer = get_model_and_tokenizer(args.model_path)

    # Get responses
    if not os.path.exists(args.save_response_path):
        responses = []
        for datapoint in tqdm(ds, desc='Generating responses'):
            response = generate_response(model, graph, noise, tokenizer, datapoint['prefix'], datapoint['suffix'],
                                        batch_size=args.batch_size, steps=args.steps)
            datapoint['response'] = response
            responses.append(datapoint)
        with open(args.save_response_path, "w") as f:
            json.dump(responses, f, indent=4)

    # Judge whether the responses are safe or not
    judge_responses(responses)


if __name__=='__main__':
    main()