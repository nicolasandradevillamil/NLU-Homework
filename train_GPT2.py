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

parser = argparse.ArgumentParser(description="Fine-tune GPT-2 for generating cooking recipes.")

# Add command-line arguments
parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
parser.add_argument('--num-train-epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--eval-steps', type=int, default=100, help='Evaluate every N steps')
parser.add_argument('--output-dir', type=str, default='GPT2-LM', help='Output directory for model checkpoints')
parser.add_argument('--context-length', type=int, default=128, help='Max length of tokenized inputs')
parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay')
parser.add_argument('--max-iters', type=int, default=500, help='Number of training steps')
parser.add_argument('--eval-iters', type=int, default=100, help='Number of evaluation steps')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
parser.add_argument('--num-warmup-steps', type=int, default=10, help='Warmup steps for learning rate scheduler')
parser.add_argument('--prompt', type=str, default=" BEGINRECIPE ", help='Prompt for text generation')
parser.add_argument('--print-frequency',type=int,default=20,help='Interval before printing loss')
parser.add_argument('--max_tokens',type=int,default=300,help='Max number of tokens generated')

args = parser.parse_args()

batch_size = args.batch_size
num_train_epochs = args.num_train_epochs
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

tokenizer = AutoTokenizer.from_pretrained('gpt2',padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id

def tokenize(element):
    outputs = tokenizer(
        element["recipe"],
        padding=True,
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

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False) #Takes care of creating batches and labels. Can be used for casual and masked language modeling.

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


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def train_model():
    model.train()
    time_start = time.time()
    for step, batch in enumerate(train_dataloader, start=1):
        if step > num_training_steps:
            break
        logits = model(batch["input_ids"]).logits
        loss = keytoken_loss(batch["input_ids"], logits)
        if step % print_frequency == 0:
            new_time = time.time()
            elapsed = new_time-time_start

            accelerator.print(
                {"Time elapsed ": str(int(elapsed//60))+"min "+ str(int(elapsed%60))+"s",
                    "Steps": str(step)+"/"+str(num_training_steps),
                    "Loss/train": round(loss.item(),4),
                }
            )
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0) #Gradient clipping
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()


        if (step % eval_steps) == 0:
            eval_loss, perplexity = evaluate()
            accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
            model.train()
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)

    eval_loss, perplexity = evaluate()
    accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)

train_model()

prompt = args.prompt
max_tokens = args.max_tokens
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_length=max_tokens,
)

generated_text = generator(prompt, num_return_sequences=1)[0]["generated_text"]