#This code will finetune GPT2 to generate cooking recipes.
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader

from datasets import load_dataset
from preprocessing import create_full_recipes
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, DataCollatorForLanguageModeling, get_scheduler, pipeline
from accelerate import Accelerator

import argparse
import time
import json

parser = argparse.ArgumentParser(description="Fine-tune GPT-2 for generating cooking recipes.")

# Add command-line arguments
parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
parser.add_argument('--eval-steps', type=int, default=100, help='Evaluate every N steps')
parser.add_argument('--output-dir', type=str, default='GPT2-LM', help='Output directory for model checkpoints')
parser.add_argument('--context-length', type=int, default=128, help='Max length of tokenized inputs')
parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay')
parser.add_argument('--max-iters', type=int, default=10000, help='Number of training steps')
parser.add_argument('--eval-iters', type=int, default=100, help='Number of evaluation steps')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
parser.add_argument('--num-warmup-steps', type=int, default=200, help='Warmup steps for learning rate scheduler')
parser.add_argument('--prompt', type=str, default=" BEGINRECIPE ", help='Prompt for text generation')
parser.add_argument('--print-frequency',type=int,default=20,help='Interval before printing loss')
parser.add_argument('--max_tokens',type=int,default=1000,help='Max number of tokens generated')
parser.add_argument('--k',type=int,default=50,help='')
parser.add_argument('--results_path', type=str, default='GPT-2.json', help='Path for the json file with results')
parser.add_argument('--train', type=int, default=1, help='1 if training model from scratch, 0 if loading a pretrained model')


args = parser.parse_args()

batch_size = args.batch_size
eval_steps = args.eval_steps
output_dir = args.output_dir
context_length = args.context_length
weight_decay = args.weight_decay
num_training_steps = args.max_iters
num_eval_steps = args.eval_iters
num_warmup_steps = args.num_warmup_steps
lr = args.lr
print_frequency = args.print_frequency

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = "cuda" if args.cuda else "cpu"

train_dataset = create_full_recipes(load_dataset("m3hrdadfi/recipe_nlg_lite",split="train"))
test_dataset = create_full_recipes(load_dataset("m3hrdadfi/recipe_nlg_lite",split="test"))

tokenizer = AutoTokenizer.from_pretrained('gpt2',padding=True,truncation=True,padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id

def tokenize(element):
    outputs = tokenizer(
        element["recipe"],
        truncation=True, 
        max_length=context_length,
        return_overflowing_tokens=True,  
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

tokenized_train = train_dataset.map(
    tokenize, batched=True, remove_columns=train_dataset.column_names
)

tokenized_test = test_dataset.map(
    tokenize, batched=True, remove_columns=test_dataset.column_names
)

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx = context_length,
    bos_token_id = tokenizer.bos_token_id,
    eos_token_id = tokenizer.eos_token_id,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

model = GPT2LMHeadModel(config)
model.to(device)

def keytoken_loss(inputs, logits):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    loss = loss_per_sample.mean()
    return loss


tokenized_train.set_format("torch")
tokenized_test.set_format("torch")

train_dataloader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(tokenized_test, batch_size=batch_size)

def get_grouped_params(model,no_decay=["bias","LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader,start=1):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
        if step > eval_steps:
            break
    loss = torch.mean(torch.FloatTensor(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()

model = GPT2LMHeadModel(config)
optimizer = AdamW(get_grouped_params(model),lr=5e-4)
accelerator = Accelerator()

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_training_steps=num_training_steps,
    num_warmup_steps=num_warmup_steps
)

data = {}
def train_model():
    model.train()
    num_epochs = (num_training_steps//len(train_dataloader))+1
    time_start = time.time()
    steps_completed = 0
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            if steps_completed > num_training_steps:
                break
            logits = model(batch["input_ids"]).logits
            loss = keytoken_loss(batch["input_ids"], logits)
            if steps_completed % print_frequency == 0:
                new_time = time.time()
                elapsed = new_time-time_start

                accelerator.print(
                    {"Time elapsed ": str(int(elapsed//60))+"min "+ str(int(elapsed%60))+"s",
                        "Steps": str(steps_completed)+"/"+str(num_training_steps),
                        "Loss/train": round(loss.item(),4),
                    }
                )
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0) #Gradient clipping
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            steps_completed+=1

            if (steps_completed % eval_steps) == 0:
                eval_loss, perplexity = evaluate()
                accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
                model.train()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(output_dir)
            
    eval_loss, perplexity = evaluate()
    data["perplexity"] = perplexity
    accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})

    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
if args.train==1:
    train_model()
    print("Trained model")
    file_path = args.results_path
    
    prompt = args.prompt
    max_tokens = args.max_tokens
    k = args.k

    model_inputs = tokenizer(str(prompt),return_tensors='pt').to(device)

    sample_output = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        top_k=k
    )
    result = tokenizer.decode(sample_output[0],skip_special_tokens=True)
    print(result)
    data["output"] = result
    # Write the dictionary to a JSON file
    with open(file_path, "w") as json_file:
        json.dump(data, json_file)
    print(f"Results saved to {file_path}")


else:
    from transformers import AutoConfig, AutoModel
    config = AutoConfig.from_pretrained(f"{output_dir}/config.json")
    model = GPT2LMHeadModel(config)
    model.to(device)
    prompt = args.prompt
    max_tokens = args.max_tokens
    k = args.k

    model_inputs = tokenizer(str(prompt),return_tensors='pt').to(device)

    sample_output = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        top_k=k
    )
    result = tokenizer.decode(sample_output[0],skip_special_tokens=True)
    print(result)

