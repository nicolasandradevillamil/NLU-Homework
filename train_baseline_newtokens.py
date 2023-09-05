#This code will train a small GPT with the GPT-2 tokenizer.
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer
import pandas as pd

from preprocessing import create_full_recipes
from baseline_model import *
from datasets import load_dataset
import time
import argparse

parser = argparse.ArgumentParser(description='Transformer decoder')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Batch size for training')
parser.add_argument('--block-size', type=int, default=128,
                    help='Batch size for training')
parser.add_argument('--max_iters', type=int, default=5000,
                    help='Number of iterations to train')
parser.add_argument('--eval_iters', type=int, default=2000,
                    help='Number of iterations to evaluate')
parser.add_argument('--lr', type=float, default=3e-4,
                    help='Initial learning rate')
parser.add_argument('--print-interval', type=float, default=20,
                    help='Number of training batches before printing loss')
parser.add_argument('--n_embed', type=float, default=384,
                    help='')
parser.add_argument('--n_head', type=float, default=6,
                    help='')
parser.add_argument('--n_layer', type=float, default=3,
                    help='')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Training dropout')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 1)')
parser.add_argument('--save_path', type=str, default="baseline_newtokens.pkl",
                    help='Path for saving the model (make sure it exists)')
parser.add_argument('--checkpoint_iters', type=int, default=1000,
                    help='Number of training iterations for saving the model')
parser.add_argument('--max_new_tokens', type=int, default=1000,
                    help='Number of training iterations for saving the model')
parser.add_argument('--context_length', type=int, default=128,
                    help='Max length of tokenized inputs')
parser.add_argument('--prompt', type=str, default='',
                    help='Prompt for the recipe generation.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

train_dataset = create_full_recipes(load_dataset("m3hrdadfi/recipe_nlg_lite",split="train"))
test_dataset = create_full_recipes(load_dataset("m3hrdadfi/recipe_nlg_lite",split="test"))

#Hyperparameters
batch_size = args.batch_size
block_size = args.block_size
context_length = args.context_length
print_interval = args.print_interval
learning_rate = args.lr
device = "cuda" if args.cuda else "cpu"
n_embed = args.n_embed
n_head = args.n_head
n_layer = args.n_layer
dropout = args.dropout
lr = args.lr
max_iters = args.max_iters
eval_iters = args.eval_iters
checkpoint_iters = args.checkpoint_iters
save_path = args.save_path
max_new_tokens = args.max_new_tokens

#TODO Implement the GPT-2 tokenizer 
#-------------------------------------------------------------
#YOUR CODE HERE
tokenizer = ''
tokenized_train = ''
tokenized_test = ''
vocab_size = ''
if tokenizer == '':
    raise NotImplementedError 
#-------------------------------------------------------------

train_dataset = pd.DataFrame(tokenized_train)
test_dataset = pd.DataFrame(tokenized_test)

train_data = torch.tensor(train_dataset['input_ids'])
test_data = torch.tensor(test_dataset['input_ids'])

def get_batch(split):
  #Generate a small batch of data of inputs x and targets y
  data = train_data.flatten() if split=="train" else test_data.flatten()
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x,y

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  losses = torch.zeros(eval_iters)
  for i in range(eval_iters):
     if i%print_interval == 0:
        print(f"Evaluating: {i}/{eval_iters}")
     X,Y = get_batch("test")
     X,Y = X.to(device), Y.to(device)
     logits, loss = model(X,Y)
     losses[i] = loss
  test_perplexity = torch.exp(losses.mean())
  out = losses.mean()
  return out,test_perplexity

class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    #each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size,n_embed)
    self.blocks = nn.Sequential(*[Block(n_embed, n_head,dropout,block_size) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embed)
    self.lm_head = nn.Linear(n_embed,vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    #idx and targets are both (B,T) tensor of integers
    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T,device=device))
    x = tok_emb + pos_emb
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) #(B,T,vocab_size)

    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits,targets)

    return logits, loss
  

  def generate(self,idx,max_new_tokens):
    # idx is (B,T) array of indices in the current context
    for _ in range(max_new_tokens):
      idx_cond = idx[:,-block_size:]
      #get the predictions
      logits, loss = self(idx_cond)
      #focus only on the last time step
      logits = logits[:,-1,:] #becomes(B,C)
      #apply softmax to get probabilites
      probs = F.softmax(logits,dim=1) #(B,C)
      #sample from the distribution
      idx_next = torch.multinomial(probs,num_samples=1) #(B,1)
      #append sampled index to the running sequence
      idx = torch.cat((idx, idx_next),dim=1) #(B,T+1)
    return idx


model = BigramLanguageModel()
m = model.to(device)

#Training
optimizer = torch.optim.AdamW(model.parameters(),lr = 1e-3)
def train_model():
        time_start = time.time()
        for i in range(max_iters):
            # sample a batch of data
            x, y = get_batch("train")
            x, y = x.to(device), y.to(device)
            
            # evaluate the loss
            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            #Print loss/save model
            if i % print_interval == 0:
                time_elapsed = time.time() - time_start
                print(f"Step: {i}/{max_iters}, Loss: {loss:.4f},\n"
                      f"Time elapsed: {int(time_elapsed//60)}min {int(time_elapsed%60)}s")
            if i % checkpoint_iters == 0 and i != 0:
               print("Saving model")
               torch.save(model.state_dict(), save_path)
        #Output loss and perplexity evaluated on the test set
        test_loss, test_perplexity = estimate_loss()
        print(f"test loss {test_loss:.4f}, test perplexity {test_perplexity:.4f}")
train_model()
torch.save(model.state_dict(), save_path)

prompt = args.prompt
if prompt == '':
    idx = torch.zeros((1,1),dtype=torch.long,device=device) #Starts up generation, single tensor with a 0
else:
    idx = ''
    #TODO Modify this part of the code so that you can write a prompt for the model
    #---------------------------------------------------------------------------------
    #YOUR CODE HERE 
    #---------------------------------------------------------------------------------
print(tokenizer.decode(model.generate(idx,max_new_tokens=max_new_tokens)[0].tolist()))
